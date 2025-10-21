class QuotaMonitor {
    constructor() {
        this.pollInterval = 5 * 60 * 1000;
        this.awsElement = document.getElementById('quotaAWS');
        this.azureElement = document.getElementById('quotaAzure');
        this.init();
    }

    async init() {
        await this.fetchQuota();
        setInterval(() => this.fetchQuota(), this.pollInterval);
    }

    async fetchQuota() {
        try {
            const response = await fetch('/api/data/quota/status');
            const data = await response.json();

            if (data.status === 'success' && data.quotas) {
                this.updateDisplay(data.quotas);
            }
        } catch (error) {
            console.warn('[QuotaMonitor] Failed to fetch quota:', error);
        }
    }

    updateDisplay(quotas) {
        if (quotas.aws) {
            const aws = quotas.aws;
            if (aws.error) {
                this.awsElement.textContent = 'AWS: unavailable';
            } else {
                this.awsElement.textContent = `AWS: ${aws.used}/${aws.total} (${aws.unit})`;
            }
        }

        if (quotas.azure) {
            const azure = quotas.azure;
            if (azure.error) {
                this.azureElement.textContent = 'Azure: unavailable';
            } else {
                this.azureElement.textContent = `Azure: ${azure.used}/${azure.total} (${azure.unit})`;
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new QuotaMonitor();
});
