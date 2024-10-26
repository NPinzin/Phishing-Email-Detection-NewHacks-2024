chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'extractEmailData') {
        extractEmailData().then(data => {
            sendResponse({ success: true, data });
        }).catch(error => {
            console.error('Error extracting email data:', error);
            sendResponse({ success: false, error: error.message });
        });
        return true; // Keep the message channel open for sendResponse
    }
});

async function extractEmailData() {
    function getPlatform() {
        const url = window.location.href;
        if (url.includes('mail.google.com')) {
            return 'gmail';
        } else if (url.includes('outlook.office.com') || url.includes('outlook.live.com')) {
            return 'outlook';
        }
        return null;
    }

    function waitForElement(selector, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const interval = 100; // Check every 100ms
            let elapsedTime = 0;

            const checkElement = setInterval(() => {
                const element = document.querySelector(selector);
                if (element) {
                    clearInterval(checkElement);
                    resolve(element);
                } else if (elapsedTime >= timeout) {
                    clearInterval(checkElement);
                    reject(new Error('Element not found: ' + selector));
                }
                elapsedTime += interval;
            }, interval);
        });
    }

    async function extractGmailContent() {
        let senderName = '';
        let senderEmail = '';
        let replyTo = '';
        let subject = '';
        let body = '';

        try {
            // Wait for the email body to load
            await waitForElement('div[role="main"]');

            // Get sender name and email using robust selectors
            const senderElement = document.querySelector('div[role="main"] [email][name]');
            if (senderElement) {
                senderName = senderElement.getAttribute('name') || senderElement.textContent || '';
                senderEmail = senderElement.getAttribute('email') || '';
            } else {
                throw new Error('Sender information not found.');
            }

            // Get reply-to address
            const replyToElement = document.querySelector('div[role="main"] [email][name]');
            if (replyToElement && replyToElement !== senderElement) {
                replyTo = replyToElement.getAttribute('email') || '';
            }

            // Get subject
            const subjectElement = document.querySelector('h2.hP');
            if (subjectElement) {
                subject = subjectElement.textContent || '';
            } else {
                throw new Error('Subject not found.');
            }

            // Get email body
            const bodyElement = document.querySelector('div.a3s');
            if (bodyElement) {
                body = bodyElement.innerHTML || bodyElement.textContent || '';
            } else {
                throw new Error('Email body not found.');
            }

            return { senderName, senderEmail, replyTo, subject, body };
        } catch (error) {
            console.error('Error extracting Gmail content:', error);
            throw error;
        }
    }

    async function extractOutlookContent() {
        let senderName = '';
        let senderEmail = '';
        let replyTo = '';
        let subject = '';
        let body = '';

        try {
            // Wait for the email body to load
            await waitForElement('div[aria-label="Message body"]');

            // Get sender info using robust selectors
            const senderElement = document.querySelector('div[aria-label="Message header"] span[role="presentation"][data-read-only="true"]');
            if (senderElement) {
                senderName = senderElement.textContent || '';
                const emailElement = senderElement.closest('div').querySelector('a[href^="mailto:"]');
                if (emailElement) {
                    senderEmail = emailElement.getAttribute('href').replace('mailto:', '') || '';
                }
            } else {
                throw new Error('Sender information not found.');
            }

            // Get subject
            const subjectElement = document.querySelector('div[aria-label="Message header"] span[data-test-id="message-subject"]');
            if (subjectElement) {
                subject = subjectElement.textContent || '';
            } else {
                throw new Error('Subject not found.');
            }

            // Get email body
            const bodyElement = document.querySelector('div[aria-label="Message body"]');
            if (bodyElement) {
                body = bodyElement.innerHTML || bodyElement.textContent || '';
            } else {
                throw new Error('Email body not found.');
            }

            return { senderName, senderEmail, replyTo, subject, body };
        } catch (error) {
            console.error('Error extracting Outlook content:', error);
            throw error;
        }
    }

    const platform = getPlatform();
    if (!platform) {
        throw new Error('Not on a supported email platform!');
    }

    if (platform === 'gmail') {
        return await extractGmailContent();
    } else if (platform === 'outlook') {
        return await extractOutlookContent();
    } else {
        throw new Error('Unsupported platform!');
    }
}
