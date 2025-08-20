document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const resumeInput = document.getElementById('resume');
    let resumeFile = null;
    let awaitingFeedback = false;
    let lastMatchResult = null;

    // Trigger file input click when upload button is clicked
    uploadBtn.addEventListener('click', () => {
        resumeInput.click();
    });

    // Handle resume file selection
    resumeInput.addEventListener('change', (e) => {
        resumeFile = e.target.files[0];
        if (resumeFile) {
            addMessage('user', `Uploaded resume: ${resumeFile.name}`);
            addMessage('bot', 'Great! Now please enter the job description.');
        }
    });

    // Handle send button click
    sendBtn.addEventListener('click', async () => {
        const message = userInput.value.trim();
        if (!message && !awaitingFeedback) return;

        if (message) {
            addMessage('user', message);
            userInput.value = '';
        }

        if (awaitingFeedback) {
            // Handle feedback submission
            const inputParts = message.split('|').map(s => s.trim());
            const removedSkills = inputParts[0] ? inputParts[0].split(',').map(s => s.trim()).filter(s => s) : [];
            const suggestedSkills = inputParts[1] ? inputParts[1].split(',').map(s => s.trim()).filter(s => s) : [];
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
            // Reset state for new match
            awaitingFeedback = false;
            resumeFile = null;
            resumeInput.value = '';
            addMessage('bot', 'Ready for a new match? Please upload a resume to start.');
        } else {
            // Handle job description submission
            if (!resumeFile) {
                addMessage('bot', 'Please upload a resume first.');
                return;
            }

            const formData = new FormData();
            formData.append('resume', resumeFile);
            formData.append('job_description', message);

            try {
                const response = await fetch('/api/match', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Error matching resume');
                }

                const data = await response.json();
                lastMatchResult = data;
                addMessage('bot', `Match found: ${data.similarity_score}% similarity with skills: ${data.matched_keywords.join(', ')}`);
                addMessage('bot', `
                    <div class="feedback-prompt">
                        <strong>Please review the extracted skills:</strong><br>
                        <strong>Resume Skills:</strong> ${data.resume_skills.join(', ') || 'None'}<br>
                        <strong>Job Description Skills:</strong> ${data.jd_skills.join(', ') || 'None'}<br>
                        <strong>Matched Skills:</strong> ${data.matched_keywords.join(', ') || 'None'}<br>
                        <em>Tip: Enter skills to remove (e.g., "hyderabad, university") and/or suggest missing skills (e.g., "backend, frontend") separated by "|". Example: "hyderabad, university|backend, frontend". Leave blank to skip.</em><br>
                        Enter skills to remove and/or suggest (format: remove1, remove2|suggest1, suggest2) or press Enter to skip.
                    </div>
                `);
                awaitingFeedback = true;
            } catch (error) {
                addMessage('bot', `Error: ${error.message}`);
            }
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
        messageDiv.className = `${sender}-message`;
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