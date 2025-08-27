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
        if (!message && !awaitingFeedback) {
            addMessage('bot', 'Please upload a resume and provide a job description.');
            return;
        }

        if (awaitingFeedback) {
            const removedSkills = message.split(',').map(s => s.trim()).filter(s => s);
            await sendFeedback(removedSkills);
            return;
        }

        if (!resumeFile) {
            addMessage('bot', 'Please upload a resume first.');
            return;
        }

        if (message) {
            addMessage('user', message);
            userInput.value = '';
            uploadProgress.style.width = '50%';

            const formData = new FormData();
            formData.append('resume', resumeFile);
            formData.append('job_description', message);

            try {
                const response = await fetch('/api/match', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to process resume and JD.');
                }

                const result = await response.json();
                lastMatchResult = result;

                if (result.error) {
                    addMessage('bot', `Error: ${result.details}`);
                    uploadProgress.style.width = '0%';
                    return;
                }

                let resultHtml = '<h3>Analysis Results</h3>';
                resultHtml += `<p><strong>Similarity Score:</strong> ${result.similarity_score.toFixed(2)}%</p>`;
                resultHtml += `<p><strong>ATS Score:</strong> ${result.ats_score.toFixed(2)}%</p>`;
                resultHtml += `<p><strong>Overall Score:</strong> ${result.overall_score.toFixed(2)}%</p>`;
                resultHtml += '<h3>Skills</h3>';
                resultHtml += '<ul>';
                resultHtml += `<li><strong>Matched Skills:</strong> ${result.matched_keywords.join(', ') || 'None'}</li>`;
                resultHtml += `<li><strong>Resume Skills:</strong> ${result.resume_skills.join(', ') || 'None'}</li>`;
                resultHtml += `<li><strong>JD Skills:</strong> ${result.jd_skills.join(', ') || 'None'}</li>`;
                resultHtml += '</ul>';
                resultHtml += '<h3>Skills by Resume Section</h3>';
                resultHtml += '<ul>';
                for (const [section, skills] of Object.entries(result.resume_skills_by_section)) {
                    resultHtml += `<li><strong>${section}:</strong> ${skills.join(', ') || 'No skills detected'}</li>`;
                }
                resultHtml += '</ul>';
                resultHtml += '<h3>ATS Issues</h3>';
                resultHtml += `<ul>${result.ats_issues.map(issue => `<li>${issue}</li>`).join('') || '<li>None</li>'}</ul>`;
                resultHtml += '<h3>Suggestions</h3>';
                resultHtml += `<ul>${result.suggestions.map(sug => `<li>${sug}</li>`).join('') || '<li>None</li>'}</ul>`;
                resultHtml += '<h3>Grammar Errors</h3>';
                resultHtml += `<ul>${result.grammar_errors.map(err => `<li>${err.message} (Context: ${err.context})</li>`).join('') || '<li>None</li>'}</ul>`;
                resultHtml += `<p><strong>Quantifiable Achievements:</strong> ${result.quantifiable_pct.toFixed(2)}%</p>`;
                resultHtml += `<p><strong>Action Verbs:</strong> ${result.action_verb_pct.toFixed(2)}%</p>`;

                addMessage('bot', resultHtml);
                uploadProgress.style.width = '0%';
                if (result.resume_skills.length > 0) {
                    addMessage('bot', 'Any skills incorrectly identified? Enter them (comma-separated) or type "no" to continue.');
                    awaitingFeedback = true;
                    userInput.placeholder = 'Enter incorrect skills (e.g., skill1, skill2) or "no"';
                } else {
                    addMessage('bot', 'No skills detected in resume. Ready for a new match? Please upload a resume to start.');
                    resumeFile = null;
                    resumeInput.value = '';
                    userInput.placeholder = 'Type your message...';
                }
            } catch (error) {
                addMessage('bot', `Error: ${error.message}`);
                uploadProgress.style.width = '0%';
            }
        }
    });

    async function sendFeedback(removedSkills) {
        if (removedSkills.length === 1 && removedSkills[0].toLowerCase() === 'no') {
            addMessage('user', 'no');
            awaitingFeedback = false;
            resumeFile = null;
            resumeInput.value = '';
            userInput.placeholder = 'Type your message...';
            addMessage('bot', 'Ready for a new match? Please upload a resume to start.');
            return;
        }
        
        const feedbackPayload = {
            removed_skills: removedSkills
        };

        try {
            const response = await fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedbackPayload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to submit feedback.');
            }

            const result = await response.json();
            addMessage('bot', result.message);
            awaitingFeedback = false;
            resumeFile = null;
            resumeInput.value = '';
            userInput.placeholder = 'Type your message...';
            addMessage('bot', 'Ready for a new match? Please upload a resume to start.');
        } catch (error) {
            addMessage('bot', `Error: ${error.message}`);
        }
    }

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
