document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const resumeInput = document.getElementById('resume');
    const uploadProgress = document.getElementById('upload-progress');
    let resumeFile = null;
    let awaitingFeedback = false;
    let lastMatchResult = null;

    // Set initial placeholder
    userInput.placeholder = 'Type your message...';

    // Trigger file input click when upload button is clicked
    uploadBtn.addEventListener('click', () => {
        resumeInput.click();
    });

    // Handle resume file selection
    resumeInput.addEventListener('change', (e) => {
        resumeFile = e.target.files[0];
        if (resumeFile) {
            uploadProgress.style.width = '100%';
            setTimeout(() => {
                uploadProgress.style.width = '0%';
                addMessage('user', `Uploaded resume: ${resumeFile.name}`);
                addMessage('bot', 'Please enter the job description.');
                userInput.placeholder = 'Type your job description';
            }, 1000);
        }
    });

    // Handle send button click
    sendBtn.addEventListener('click', async () => {
        const message = userInput.value.trim();
        if (!message && !awaitingFeedback) return;

        if (message && !awaitingFeedback) {
            addMessage('user', message);
            userInput.value = '';
            try {
                const formData = new FormData();
                formData.append('resume', resumeFile);
                formData.append('job_description', message);
                const response = await fetch('/api/match', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) throw new Error('Error analyzing resume');
                const data = await response.json();
                lastMatchResult = data;

                let botMessage = `
                    <strong>Resume Analysis Results</strong><br><br>
                    <strong>Skill Match:</strong> ${data.similarity_score}% with matched skills: ${data.matched_keywords.join(', ') || 'None'}<br>
                    <strong>ATS Score:</strong> ${data.ats_score}%<br>
                    <strong>Overall Resume Score:</strong> ${data.overall_score}%<br>
                    <strong>Quantifiable Achievements:</strong> ${data.quantifiable_pct}% of bullet points have numbers/metrics<br>
                    <strong>Action Verb Usage:</strong> ${data.action_verb_pct}% of words are action verbs<br>
                    <strong>Repeated Words:</strong> ${Object.entries(data.repeated_words).map(([word, freq]) => `${word} (${freq} times)`).join(', ') || 'None'}<br>
                    <strong>Buzzwords Found:</strong> ${data.buzzwords_found.join(', ') || 'None'}<br>
                    <strong>Filler Words Found:</strong> ${data.filler_found.join(', ') || 'None'}<br>
                `;
                if (data.grammar_errors.length > 0) {
                    botMessage += `<strong>Grammar/Spelling Errors:</strong><ul>${data.grammar_errors.map(err => `<li>${err.message} in '${err.context.slice(0, 50)}...' (suggestions: ${err.replacements.join(', ')})</li>`).join('')}</ul><br>`;
                }
                if (data.ats_issues.length > 0) {
                    botMessage += `<strong>ATS Issues:</strong><ul>${data.ats_issues.map(s => `<li>${s}</li>`).join('')}</ul><br>`;
                }
                if (data.ats_suggestions.length > 0) {
                    botMessage += `<strong>ATS Suggestions:</strong><ul>${data.ats_suggestions.map(s => `<li>${s}</li>`).join('')}</ul><br>`;
                }
                if (data.suggestions.length > 0) {
                    botMessage += `<strong>General Suggestions:</strong><ul>${data.suggestions.map(s => `<li>${s}</li>`).join('')}</ul><br>`;
                }
                botMessage += `<strong>Rewrite Suggestions for Bullet Points:</strong><ul>${data.rewrite_suggestions.map(s => `<li>${s}</li>`).join('')}</ul><br>`;
                botMessage += `<strong>Suggested Hard Skills:</strong> ${data.hard_skills_suggestions.join(', ') || 'None'}<br>`;
                botMessage += `<strong>Suggested Soft Skills:</strong> ${data.soft_skills_suggestions.join(', ') || 'None'}<br>`;
                addMessage('bot', botMessage);
                addMessage('bot', `
                    <div class="feedback-prompt">
                        <strong>Review Extracted Skills:</strong><br>
                        <strong>Resume Skills:</strong> ${data.resume_skills.join(', ') || 'None'}<br>
                        <strong>Job Description Skills:</strong> ${data.jd_skills.join(', ') || 'None'}<br>
                        <strong>Matched Skills:</strong> ${data.matched_keywords.join(', ') || 'None'}<br>
                        <em>Tip: Enter skills to remove (e.g., "hyderabad, university") and/or suggest missing skills (e.g., "python, sql") separated by "|". Example: "hyderabad, university|python, sql". Leave blank to skip.</em><br>
                        Type your feedback for skills to remove and/or suggest (format: remove1, remove2|suggest1, suggest2) or press Enter to skip.
                    </div>
                `);
                userInput.placeholder = 'Type your feedback';
                awaitingFeedback = true;
            } catch (error) {
                addMessage('bot', `Error: ${error.message}`);
            }
        } else if (awaitingFeedback) {
            const inputParts = message.split('|').map(s => s.trim());
            const removedSkills = inputParts[0] ? inputParts[0].split(',').map(s => s.trim()).filter(s => s) : [];
            const suggestedSkills = inputParts[1] ? inputParts[1].split(',').map(s => s.trim()).filter(s => s) : [];
            userInput.value = '';
            if (removedSkills.length > 0 || suggestedSkills.length > 0) {
                try {
                    const response = await fetch('/api/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ removed_skills: removedSkills, suggested_skills: suggestedSkills })
                    });
                    if (!response.ok) throw new Error('Error submitting feedback');
                    const data = await response.json();
                    addMessage('bot', data.message);
                } catch (error) {
                    addMessage('bot', `Error: ${error.message}`);
                }
            } else {
                addMessage('bot', 'No skills removed or suggested.');
            }
            // Reset state completely to allow new resume upload
            awaitingFeedback = false;
            resumeFile = null;
            resumeInput.value = ''; // Clear file input to allow new upload
            userInput.placeholder = 'Type your message...';
            addMessage('bot', 'Ready for a new match? Please upload a resume to start.');
        }
    });

    // Handle Enter key press
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    // Function to add messages to chat window
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message mb-4`;
        const avatarSpan = document.createElement('span');
        avatarSpan.className = 'avatar';
        avatarSpan.textContent = sender === 'bot' ? '🤖' : '👤';
        const textSpan = document.createElement('span');
        textSpan.className = 'message-text';
        textSpan.innerHTML = text;
        messageDiv.appendChild(avatarSpan);
        messageDiv.appendChild(textSpan);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});