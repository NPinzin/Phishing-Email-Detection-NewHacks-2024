// Function to determine the active email platform
function getPlatform() {
    if (window.location.href.includes('mail.google.com')) {
        return 'gmail';
    } else if (window.location.href.includes('outlook.office.com') || window.location.href.includes('outlook.live.com')) {
        return 'outlook';
    } else if (window.location.href.includes('mail.yahoo.com')) {
        return 'yahoo';
    }
    return null;
}

// Function to extract email content based on the detected platform
function extractEmailContent() {
    let platform = getPlatform();
    let emailContent = '';

    switch (platform) {
        case 'gmail':
            let gmailElement = document.querySelector('.email-body');
            if (gmailElement) emailContent = gmailElement.innerText;
            break;

        case 'outlook':
            let outlookElement = document.querySelector('.wide-content-host');
            if (outlookElement) emailContent = outlookElement.innerText;
            break;

        case 'yahoo':
            let yahooElement = document.querySelector('.email-body-container');
            if (yahooElement) emailContent = yahooElement.innerText;
            break;
    }
    return emailContent;
}

// Function to analyze email content by sending it to the backend
function analyzeEmail() {
    let emailContent = extractEmailContent();
    if (emailContent) {
        // Make an API call to the backend
        fetch('http://127.0.0.1:8000/analyze-email/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content: emailContent })
        })
            .then(response => response.json())
            .then(data => {
                if (data.phishing_probability > 0.7) {
                    alert('Warning: This email may be a phishing attempt.');
                }
            })
            .catch(error => console.error('Error:', error));
    }
}


// Add event listener to trigger email analysis when an email is opened
document.addEventListener('click', (event) => {
    if (event.target.matches('.email-subject')) {
        setTimeout(analyzeEmail, 1000); // Delay to allow content to load
    }
});
