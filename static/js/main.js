document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('scraperForm');
    const resultsContainer = document.getElementById('results');
    const progressBar = document.querySelector('.progress-bar');
    const resultContent = document.getElementById('resultContent');
    const downloadLink = document.getElementById('downloadLink');

    form.addEventListener('submit', async(e) => {
        e.preventDefault();

        // Show results container and reset progress
        resultsContainer.style.display = 'block';
        progressBar.style.width = '0%';
        resultContent.textContent = 'Starting scraping process...';
        downloadLink.style.display = 'none';

        // Get form data
        const formData = new FormData(form);
        const data = {
            url: formData.get('url'),
            prompt: formData.get('prompt'),
            output: formData.get('output'),
            crawl_detail: formData.get('crawl_detail') === 'on'
        };

        try {
            // Start scraping process
            const response = await fetch('/api/scrape', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Scraping failed');
            }

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            if (data.progress) {
                                progressBar.style.width = `${data.progress}%`;
                            }
                            if (data.message) {
                                resultContent.textContent = data.message;
                            }
                            if (data.result) {
                                result = data.result;
                            }
                        } catch (e) {
                            console.error('Error parsing chunk:', e);
                        }
                    }
                }
            }

            // Show download link if we have results
            if (result) {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                downloadLink.href = URL.createObjectURL(blob);
                downloadLink.download = data.output;
                downloadLink.style.display = 'inline-block';
            }

        } catch (error) {
            console.error('Error:', error);
            resultContent.textContent = `Error: ${error.message}`;
            progressBar.style.width = '0%';
        }
    });
});