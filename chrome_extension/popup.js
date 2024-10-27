// popup.js

document.addEventListener('DOMContentLoaded', () => {
  // Get references to elements
  const darkModeCheckbox = document.getElementById('darkMode');
  const autoScanCheckbox = document.getElementById('autoScan');
  const enableNotificationsCheckbox = document.getElementById('enableNotifications');
  const outputDiv = document.getElementById('output');
  const body = document.body;

  // Progress bar elements
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');

  // API Base URL
  const API_BASE_URL = 'http://localhost:3000';
  const API_BASE_URL2 = 'http://localhost:5000';


  // Initially check login state and toggle UI accordingly
  chrome.storage.sync.get(['isLoggedIn'], data => {
    if (data.isLoggedIn) {
      document.getElementById('auth-container').style.display = 'none';
      document.getElementById('main-content').style.display = 'block';
    } else {
      document.getElementById('auth-container').style.display = 'block';
      document.getElementById('main-content').style.display = 'none';
    }
  });

  function updateProgressBar(predictionMessage) {
    // Convert to percentage and format to 2 decimal points
    const percentage = (predictionMessage * 1000).toFixed(2);

    progressFill.style.width = percentage + '%';
    progressText.textContent = percentage + '%';
    // Change the color of the progress fill based on the percentage ranges
    if (percentage < 25) {
      progressFill.style.backgroundColor = '#006400'; // Darker green
    } else if (percentage < 50) {
      progressFill.style.backgroundColor = '#008000'; // Green
    } else if (percentage < 70) {
      progressFill.style.backgroundColor = '#f1c40f'; // Yellow
    } else if (percentage < 90) {
      progressFill.style.backgroundColor = '#e67e22'; // Orange
    } else {
      progressFill.style.backgroundColor = '#e74c3c'; // Red
    }
  }


  // Load settings from storage
  chrome.storage.sync.get(['darkMode', 'autoScan', 'enableNotifications'], data => {
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
    chrome.tabs.create({ url: 'https://findingnemo.co' });
  });



  // Function to scan emails
  function scanEmails() {
    outputDiv.innerHTML = 'Scanning...';
    progressContainer.style.display = 'block';
    updateProgressBar(0);

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs.length === 0) {
        outputDiv.innerHTML = '<p style="color:red;">No active tab found.</p>';
        progressContainer.style.display = 'none';
        return;
      }

      chrome.tabs.sendMessage(tabs[0].id, { action: 'extractEmailData' }, (response) => {
        if (chrome.runtime.lastError) {
          console.error('Runtime error:', chrome.runtime.lastError.message);
          outputDiv.innerHTML = `<p style="color:red;">Error: ${chrome.runtime.lastError.message}</p>`;
          progressContainer.style.display = 'none';
          return;
        }
        if (response && response.success) {
          const data = response.data;
          let message = {
            senderName: data.senderName || 'N/A',
            senderEmail: data.senderEmail || 'N/A',
            replyTo: data.replyTo || 'N/A',
            subject: data.subject || 'N/A',
            body: data.body || 'N/A',
          };
          outputDiv.innerHTML = `<pre>${JSON.stringify(message, null, 2)}</pre>`;

          // Send the data to the server
          fetch('http://localhost:5000/send-string', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: data.body }),
          })
            .then((response) => response.json())
            .then((data) => {
              console.log('Data received from Python scraper:', data);
              if (data.prediction !== undefined) {
                // Update progress bar based on risk percentage
                updateProgressBar(data.prediction);

                // Display the prediction result to the user
                const predictionMessage = data.prediction;
                outputDiv.innerHTML += `<p>${predictionMessage}</p>`;

                // Now send the email data and prediction to our server
                chrome.storage.sync.get('userEmail', ({ userEmail }) => {
                  if (userEmail) {
                    fetch(`${API_BASE_URL}/api/save-email-verification`, {
                      method: 'POST',
                      headers: {
                        'Content-Type': 'application/json',
                      },
                      body: JSON.stringify({
                        userEmail: userEmail,
                        emailData: message,
                        prediction: data.prediction,
                      }),
                    })
                      .then((response) => response.json())
                      .then((saveData) => {
                        console.log('Email verification data saved:', saveData);
                      })
                      .catch((error) => {
                        console.error('Error saving email verification data:', error);
                      });
                  } else {
                    console.error('User email not found in storage.');
                  }
                });

              } else {
                outputDiv.innerHTML += `<p style="color:red;">Error: ${data.message}</p>`;
                progressContainer.style.display = 'none';
              }
            })
            .catch((error) => {
              console.error('Error sending data to Python scraper:', error);
              outputDiv.innerHTML += `<p style="color:red;">Error sending data to the server: ${error.message}</p>`;
              progressContainer.style.display = 'none';
            });

          // Show notification if enabled
          chrome.storage.sync.get('enableNotifications', (storageData) => {
            if (storageData.enableNotifications) {
              chrome.notifications.create('scanComplete', {
                type: 'basic',
                iconUrl: 'icon128.png',
                title: 'Scam Detection',
                message: 'Scan completed successfully.',
                buttons: [{ title: 'View Details' }],
              });
            }
          });
        } else {
          const errorMsg = response && response.error ? response.error : 'Unknown error';
          console.error('Error in response:', errorMsg);
          outputDiv.innerHTML = `<p style="color:red;">Error extracting data: ${errorMsg}</p>`;
          progressContainer.style.display = 'none';
        }
      });
    });
  }



  // Handle Tab Switching in Auth Forms
  const tablinks = document.getElementsByClassName('tablinks');
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].addEventListener('click', function () {
      openTab(this.textContent.trim());
    });
  }


  function openTab(tabName) {
    const tabcontent = document.getElementsByClassName('tabcontent');
    const tablinks = document.getElementsByClassName('tablinks');

    for (let i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = 'none';
    }

    for (let i = 0; i < tablinks.length; i++) {
      tablinks[i].classList.remove('active');
      if (tablinks[i].textContent.trim() === tabName) {
        tablinks[i].classList.add('active');
      }
    }

    document.getElementById(tabName).style.display = 'block';
  }


  // Handle Login Form Submission
  const loginForm = document.getElementById('login-form');
  loginForm.addEventListener('submit', async e => {
    e.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    try {
      const response = await fetch(`${API_BASE_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();
      if (response.ok) {
        chrome.storage.sync.set({ isLoggedIn: true, userEmail: email }, () => {
          document.getElementById('auth-container').style.display = 'none';
          document.getElementById('main-content').style.display = 'block';
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();
      if (response.ok) {
        alert('Signup successful! You can now log in.');
        document.getElementById('login-email').value = email;
        document.getElementById('login-password').value = password;
        openTab('Login');
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
      document.getElementById('main-content').style.display = 'none';
      alert('Logged out successfully.');
    });
  });
});
