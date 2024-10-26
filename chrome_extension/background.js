// Listener for when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
    console.log("Phishing Detection Extension installed.");
  });
  
  // Event listener for tab updates
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    // Check if the tab URL matches supported email platforms
    if (
      (tab.url.includes("mail.google.com") ||
      tab.url.includes("outlook.office.com") ||
      tab.url.includes("outlook.live.com") ||
      tab.url.includes("mail.yahoo.com")) &&
      changeInfo.status === "complete"
    ) {
      // Inject the content script
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ["content.js"]
      }).catch((error) => console.error('Failed to inject script:', error));
    }
  });
  
