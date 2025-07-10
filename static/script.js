// Simple JavaScript for stock analysis tool

// Copy to clipboard functionality
function copyToClipboard() {
    const resultText = document.querySelector('.analysis-result pre').textContent;
    
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(resultText).then(() => {
            showNotification('Results copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            fallbackCopy(resultText);
        });
    } else {
        fallbackCopy(resultText);
    }
}

// Fallback copy method for older browsers
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showNotification('Results copied to clipboard!');
    } catch (err) {
        console.error('Fallback copy failed: ', err);
        showNotification('Copy failed. Please select and copy manually.');
    }
    
    document.body.removeChild(textArea);
}

// Show notification
function showNotification(message) {
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #27ae60;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 1000;
        font-family: inherit;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Form validation
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.analysis-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const ticker = document.getElementById('ticker').value.trim();
            const question = document.getElementById('question').value.trim();
            
            if (!ticker) {
                e.preventDefault();
                showNotification('Please enter a stock symbol');
                return;
            }
            
            if (!question) {
                e.preventDefault();
                showNotification('Please enter a question');
                return;
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            submitBtn.textContent = 'Analyzing...';
            submitBtn.disabled = true;
        });
    }
});

// Auto-uppercase stock ticker
document.addEventListener('DOMContentLoaded', function() {
    const tickerInput = document.getElementById('ticker');
    if (tickerInput) {
        tickerInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
    }
});