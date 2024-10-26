document.getElementById('scanEmails').addEventListener('click', () => {
  // Clear previous output
  const outputDiv = document.getElementById('output');
  outputDiv.innerHTML = 'Scanning...';

  // Send a message to the content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length === 0) {
      outputDiv.innerHTML = '<p style="color:red;">No active tab found.</p>';
      return;
    }

    chrome.tabs.sendMessage(tabs[0].id, { action: 'extractEmailData' }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('Runtime error:', chrome.runtime.lastError.message);
        outputDiv.innerHTML = `<p style="color:red;">Error: ${chrome.runtime.lastError.message}</p>`;
        return;
      }
      if (response && response.success) {
        const data = response.data;
        // Display the extracted data
        let message = `<strong>Sender Name:</strong> ${data.senderName || 'N/A'}<br>`;
        message += `<strong>Sender Email:</strong> ${data.senderEmail || 'N/A'}<br>`;
        message += `<strong>Reply-To:</strong> ${data.replyTo || 'N/A'}<br>`;
        message += `<strong>Subject:</strong> ${data.subject || 'N/A'}<br><br>`;
        message += `<strong>Email Content:</strong><br><pre>${data.body || 'N/A'}</pre>`;
        outputDiv.innerHTML = message;
      } else {
        const errorMsg = response && response.error ? response.error : 'Unknown error';
        console.error('Error in response:', errorMsg);
        outputDiv.innerHTML = `<p style="color:red;">Error extracting data: ${errorMsg}</p>`;
      }
    });
  });
});

document.getElementById('visitWebsite').addEventListener('click', () => {
  // Open the website in a new tab
  chrome.tabs.create({ url: 'https://youtube.com' }); // Replace with your actual website URL
});
