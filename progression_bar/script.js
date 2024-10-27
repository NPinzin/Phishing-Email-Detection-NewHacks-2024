document.addEventListener('DOMContentLoaded', () => {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // Set the percentage directly here
    const percentage = 80; // Set your desired percentage value here

    // Initialize the progress bar based on the percentage
    updateProgress(percentage);

    // Function to update the progress bar and gradient color
    function updateProgress(percentage) {
        // Ensure percentage is between 0 and 100
        if (percentage < 0 || percentage > 100 || isNaN(percentage)) {
            alert("Please enter a valid percentage (0-100)");
            return;
        }

        // Update the progress bar width and text
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `${percentage.toFixed(1)}%`; // Display one decimal place

        // Calculate color based on percentage
        const gradientColor = calculateGradientColor(percentage);

        // Apply gradient based on percentage
        progressBar.style.background = `linear-gradient(to right, ${gradientColor}, ${gradientColor})`;
    }

    // Function to calculate gradient color based on percentage
    function calculateGradientColor(percentage) {
        let color;

        if (percentage <= 25) {
            // 0% to 25%: Dark Red to Orange
            const ratio = percentage / 25;
            const red = Math.round(139 + (255 - 139) * ratio);   // Dark Red (139, 0, 0) to Orange (255, 165, 0)
            const green = Math.round(0 + 165 * ratio);            // Green value from 0 to 165
            color = `rgb(${red}, ${green}, 0)`;
        } else if (percentage <= 50) {
            // 26% to 50%: Orange to Yellow
            const ratio = (percentage - 25) / 25;
            const red = 255;                                      // Constant at 255
            const green = Math.round(165 + (255 - 165) * ratio);  // Green value from 165 to 255
            color = `rgb(${red}, ${green}, 0)`;
        } else if (percentage <= 75) {
            // 51% to 75%: Yellow to Green
            const ratio = (percentage - 50) / 25;
            const red = Math.round(255 - 255 * ratio);            // Red decreases from 255 to 0
            const green = 255;                                    // Constant at 255
            color = `rgb(${red}, ${green}, 0)`;
        } else {
            // 76% to 100%: Green to Light Green
            const ratio = (percentage - 75) / 25;
            const green = Math.round(255 - (255 - 150) * ratio);  // Green decreases from 255 to 150
            color = `rgb(0, ${green}, 0)`; // Lighter green
        }

        return color;
    }
});
