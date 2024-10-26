document.getElementById('scanEmails').addEventListener('click', () => {
    // Logic to trigger scanning of emails
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        function: analyzeEmail
      });
    });
  });
  