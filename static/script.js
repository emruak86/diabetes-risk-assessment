document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');

    predictionForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        // Show a loading state while waiting for the server
        resultContainer.innerHTML = '<p class="loading">Calculating your risk...</p>';
        resultContainer.className = 'result-display'; // Reset classes

        const formData = new FormData(predictionForm);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'An error occurred while predicting.');
            }

            const result = await response.json();
            displayResult(result);

        } catch (error) {
            resultContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            resultContainer.classList.add('error-display');
        }
    });

    function displayResult(result) {
        const { risk_percent, risk_level } = result;
        let riskClass = '';

        // Determine the CSS class based on the risk level for color-coding
        switch (risk_level) {
            case 'Low Risk':
                riskClass = 'low-risk';
                break;
            case 'Moderate Risk':
                riskClass = 'moderate-risk';
                break;
            case 'High Risk':
                riskClass = 'high-risk';
                break;
            case 'Very High Risk':
                riskClass = 'very-high-risk';
                break;
        }

        // Populate the result container with the prediction data
        resultContainer.innerHTML = `
            <h2>Prediction Result</h2>
            <p class="risk-percentage">Your estimated diabetes risk is <span>${risk_percent}%</span></p>
            <p class="risk-level">This falls into the <strong>${risk_level}</strong> category.</p>
        `;
        resultContainer.className = `result-display ${riskClass}`;
    }
});
