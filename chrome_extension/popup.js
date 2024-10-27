document.addEventListener('DOMContentLoaded', () => {
  // Get references to elements
  const darkModeCheckbox = document.getElementById('darkMode');
  const autoScanCheckbox = document.getElementById('autoScan');
  const enableNotificationsCheckbox = document.getElementById('enableNotifications');
  const outputDiv = document.getElementById('output');
  const body = document.body;

  // API Base URL
  const API_BASE_URL = 'http://localhost:3000'; // Adjust as needed

  // Initially check login state and toggle UI accordingly
  chrome.storage.sync.get(['isLoggedIn'], (data) => {
    if (data.isLoggedIn) {
      document.getElementById('auth-container').style.display = 'none';
      document.querySelector('.container').style.display = 'block';
    } else {
      document.getElementById('auth-container').style.display = 'block';
      document.querySelector('.container').style.display = 'none';
    }
  });

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

  // Event listeners for settings toggles
  darkModeCheckbox.addEventListener('change', () => {
    if (darkModeCheckbox.checked) {
      body.classList.add('dark-mode');
    } else {
      body.classList.remove('dark-mode');
    }
    chrome.storage.sync.set({ darkMode: darkModeCheckbox.checked });
  });

  autoScanCheckbox.addEventListener('change', () => {
    chrome.storage.sync.set({ autoScan: autoScanCheckbox.checked });
    enableNotificationsCheckbox.disabled = !autoScanCheckbox.checked;

    if (!autoScanCheckbox.checked) {
      enableNotificationsCheckbox.checked = false;
      chrome.storage.sync.set({ enableNotifications: false });
    }

    if (autoScanCheckbox.checked) {
      scanEmails();
    }
  });

  enableNotificationsCheckbox.addEventListener('change', () => {
    chrome.storage.sync.set({ enableNotifications: enableNotificationsCheckbox.checked });
  });

  // Scan emails when the "Scan Emails" button is clicked
  document.getElementById('scanEmails').addEventListener('click', scanEmails);

  // Open a new tab when the "Visit Website" button is clicked
  document.getElementById('visitWebsite').addEventListener('click', () => {
    chrome.tabs.create({ url: 'https://youtube.com' });
  });

  // Function to scan emails
  function scanEmails() {
    outputDiv.innerHTML = 'Scanning...';

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
          let message = `<strong>Sender Name:</strong> ${data.senderName || 'N/A'}<br>`;
          message += `<strong>Sender Email:</strong> ${data.senderEmail || 'N/A'}<br>`;
          message += `<strong>Reply-To:</strong> ${data.replyTo || 'N/A'}<br>`;
          message += `<strong>Subject:</strong> ${data.subject || 'N/A'}<br><br>`;
          message += `<strong>Email Content:</strong><br><pre>${data.body || 'N/A'}</pre>`;
          outputDiv.innerHTML = message;

          // Send the data to the Python scraper
          fetch('http://localhost:5000/process_email', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
          })
            .then(response => response.json())
            .then(data => {
              console.log('Data sent to Python scraper:', data);
              // Optionally, display a success message to the user
              outputDiv.innerHTML += '<p style="color:green;">Email data sent to the server successfully.</p>';
            })
            .catch(error => {
              console.error('Error sending data to Python scraper:', error);
              outputDiv.innerHTML += `<p style="color:red;">Error sending data to the server: ${error.message}</p>`;
            });


          // Show notification if enabled
          chrome.storage.sync.get('enableNotifications', (storageData) => {
            if (storageData.enableNotifications) {
              chrome.notifications.create('', {
                type: 'basic',
                iconUrl: 'icon128.png',
                title: 'Phishing Detection',
                message: 'Scan completed successfully.',
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

  // Handle Login Form Submission
  const loginForm = document.getElementById('login-form');
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    try {
      const response = await fetch(`${API_BASE_URL}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      if (response.ok) {
        // Save login state in chrome.storage
        chrome.storage.sync.set({ isLoggedIn: true, userEmail: email }, () => {
          document.getElementById('auth-container').style.display = 'none';
          document.querySelector('.container').style.display = 'block';
          alert('Login successful!');
        });
      } else {
        alert(`Login failed: ${data.message}`);
      }
    } catch (error) {
      console.error('Error during login:', error);
      alert('An error occurred during login.');
    }
  });

  // Handle Signup Form Submission
  const signupForm = document.getElementById('signup-form');
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;

    try {
      const response = await fetch(`${API_BASE_URL}/api/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      if (response.ok) {
        alert('Signup successful! You can now log in.');
        document.getElementById('login-email').value = email;
        document.getElementById('login-password').value = password;
      } else {
        alert(`Signup failed: ${data.message}`);
      }
    } catch (error) {
      console.error('Error during signup:', error);
      alert('An error occurred during signup.');
    }
  });

  // Add logout functionality
  document.getElementById('logout').addEventListener('click', () => {
    chrome.storage.sync.set({ isLoggedIn: false, userEmail: null }, () => {
      document.getElementById('auth-container').style.display = 'block';
      document.querySelector('.container').style.display = 'none';
      alert('Logged out successfully.');
    });
  });
});
