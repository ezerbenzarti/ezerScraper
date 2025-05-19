document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('scraperForm');
    const results = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    const downloadLink = document.getElementById('downloadLink');
    const progressBar = document.querySelector('.progress-bar');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Show progress
        results.style.display = 'block';
        resultContent.innerHTML = 'Starting scraping process...';
        progressBar.style.width = '10%';

        // Get form data
        const formData = {
            url: document.getElementById('url').value,
            prompt: document.getElementById('prompt').value,
            crawl_detail: document.getElementById('crawlDetail').checked,
            scraping_method: document.querySelector('input[name="scraping_method"]:checked').value,
            workflow_id: Date.now().toString() // Generate a unique workflow ID
        };

        try {
            // Start scraping
            const response = await fetch('/api/scrape', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.status === 'started') {
                resultContent.innerHTML = `${data.message}<br>Workflow ID: ${data.workflow_id}`;
                progressBar.style.width = '20%';

                // Poll for results
                pollResults(data.workflow_id);
            } else {
                resultContent.innerHTML = 'Error: ' + data.message;
                progressBar.style.width = '0%';
            }
        } catch (error) {
            resultContent.innerHTML = 'Error: ' + error.message;
            progressBar.style.width = '0%';
        }
    });

    async function pollResults(workflowId) {
        try {
            const response = await fetch(`/api/data/raw?workflow_id=${workflowId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                resultContent.innerHTML = `Scraping completed!<br>Found ${data.results.length} results`;
                progressBar.style.width = '100%';

                // Enable download link
                downloadLink.href = `/api/data/raw/csv/${workflowId}`;
                downloadLink.style.display = 'block';
                downloadLink.download = document.getElementById('outputFile').value;
            } else if (data.status === 'failed') {
                resultContent.innerHTML = 'Scraping failed: ' + data.error;
                progressBar.style.width = '0%';
            } else {
                // Still processing, update progress and poll again
                progressBar.style.width = '50%';
                setTimeout(() => pollResults(workflowId), 2000);
            }
        } catch (error) {
            resultContent.innerHTML = 'Error polling results: ' + error.message;
            progressBar.style.width = '0%';
        }
    }
});