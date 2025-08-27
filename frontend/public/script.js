document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const resumeInput = document.getElementById('resume');
    const uploadProgress = document.getElementById('upload-progress');
    const loading = document.getElementById('loading');
    let resumeFile = null;

    uploadBtn.addEventListener('click', () => resumeInput.click());

    resumeInput.addEventListener('change', (e) => {
        resumeFile = e.target.files[0];
        if (resumeFile) {
            uploadProgress.style.width = '100%';
            setTimeout(() => {
                uploadProgress.style.width = '0%';
                addMessage('user', `Uploaded resume: ${resumeFile.name}`);
                addMessage('bot', 'Please enter the job description.');
            }, 1000);
        }
    });

    sendBtn.addEventListener('click', async () => {
        const message = userInput.value.trim();
        if (!resumeFile) {
            addMessage('bot', 'Please upload a resume first.');
            return;
        }
        if (!message) {
            addMessage('bot', 'Please provide a job description.');
            return;
        }

        addMessage('user', message);
        userInput.value = '';
        loading.classList.remove('hidden');
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
                throw new Error(errorData.detail || 'Failed to process resume.');
            }

            const result = await response.json();
            uploadProgress.style.width = '0%';
            loading.classList.add('hidden');

            let sectionText = '<h3>Resume Sections</h3><ul>';
            for (const [section, content] of Object.entries(result.resume_sections)) {
                if (content) {
                    sectionText += `<li><strong>${section.toUpperCase()}</strong>: ${content}</li>`;
                }
            }
            sectionText += '</ul>';
            addMessage('bot', sectionText);

            addMessage('bot', `<h3>Resume Skills</h3><p>${result.resume_skills.join(', ') || 'None'}</p>`);
            addMessage('bot', `<h3>JD Skills</h3><p>${result.jd_skills.join(', ') || 'None'}</p>`);
            addMessage('bot', `<h3>Matched Skills</h3><p>${result.matched_skills.join(', ') || 'None'}</p>`);
            addMessage('bot', `<h3>Match Percentage</h3><p>${result.match_percentage}%</p>`);

            resumeFile = null;
            resumeInput.value = '';
            addMessage('bot', 'Ready for a new match? Please upload a resume.');
        } catch (error) {
            uploadProgress.style.width = '0%';
            loading.classList.add('hidden');
            addMessage('bot', `Error: ${error.message}`);
        }
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendBtn.click();
    });

    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message animate-fade-in`;
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
