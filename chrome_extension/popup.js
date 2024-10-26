document.addEventListener('DOMContentLoaded', () => {
  // Get references to elements
  const darkModeCheckbox = document.getElementById('darkMode');
  const autoScanCheckbox = document.getElementById('autoScan');
  const enableNotificationsCheckbox = document.getElementById('enableNotifications');
  const outputDiv = document.getElementById('output');
  const body = document.body;

  // Load settings from storage
  chrome.storage.sync.get(['darkMode', 'autoScan', 'enableNotifications'], (data) => {
    darkModeCheckbox.checked = data.darkMode || false;
    autoScanCheckbox.checked = data.autoScan || false;
    enableNotificationsCheckbox.checked = data.enableNotifications || false;

    // Apply dark mode if enabled
    if (darkModeCheckbox.checked) {
      body.classList.add('dark-mode');
    }

    // Disable notifications checkbox if autoScan is not enabled
    enableNotificationsCheckbox.disabled = !autoScanCheckbox.checked;

    // If autoScan is enabled, trigger scan
    if (autoScanCheckbox.checked) {
      scanEmails();
    }
  });

  // Event listener for dark mode toggle
  darkModeCheckbox.addEventListener('change', () => {
    if (darkModeCheckbox.checked) {
      body.classList.add('dark-mode');
    } else {
      body.classList.remove('dark-mode');
    }
    // Save setting
    chrome.storage.sync.set({ darkMode: darkModeCheckbox.checked });
  });

  // Event listener for autoScan toggle
  autoScanCheckbox.addEventListener('change', () => {
    // Save setting
    chrome.storage.sync.set({ autoScan: autoScanCheckbox.checked });

    // Disable or enable notifications checkbox
    enableNotificationsCheckbox.disabled = !autoScanCheckbox.checked;

    // Uncheck notifications if autoScan is disabled
    if (!autoScanCheckbox.checked) {
      enableNotificationsCheckbox.checked = false;
      chrome.storage.sync.set({ enableNotifications: false });
    }

    // If autoScan is enabled, trigger scan
    if (autoScanCheckbox.checked) {
      scanEmails();
    }
  });

  // Event listener for enableNotifications toggle
  enableNotificationsCheckbox.addEventListener('change', () => {
    // Save setting
    chrome.storage.sync.set({ enableNotifications: enableNotificationsCheckbox.checked });
  });

  // Event listener for scanEmails button
  document.getElementById('scanEmails').addEventListener('click', scanEmails);

  // Event listener for visitWebsite button
  document.getElementById('visitWebsite').addEventListener('click', () => {
    // Open the website in a new tab
    chrome.tabs.create({ url: 'https://youtube.com' });
  });

  function scanEmails() {
    // Clear previous output
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

          // If notifications are enabled, show a notification
          chrome.storage.sync.get('enableNotifications', (data) => {
            if (data.enableNotifications) {
              chrome.notifications.create('', {
                type: 'basic',
                iconUrl: 'icon128.png',
                title: 'Phishing Detection',
                message: 'Scan completed successfully.'
              });
            }
          });
        } else {
          const errorMsg = response && response.error ? response.error : 'Unknown error';
          console.error('Error in response:', errorMsg);
          outputDiv.innerHTML = `<p style="color:red;">Error extracting data: ${errorMsg}</p>`;
        }
      });
    });
  }
});

