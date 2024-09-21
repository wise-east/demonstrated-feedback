document.addEventListener('DOMContentLoaded', function() {
    const jsonContainer = document.getElementById('json-container');
    let currentIndex = 0;
    let jsonItems = [];

    function createCollapsible(key, value) {
        const button = document.createElement('button');
        button.className = 'collapsible btn btn-light btn-sm w-100 text-start';
        button.textContent = `${key}: ${typeof value === 'object' ? '{...}' : '[...]'}`;

        const content = document.createElement('div');
        content.className = 'content';
        const pre = document.createElement('pre');
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.wordWrap = 'break-word';
        pre.textContent = JSON.stringify(value, null, 2);
        content.appendChild(pre);

        button.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            if (content.style.display === 'block') {
                content.style.display = 'none';
            } else {
                content.style.display = 'block';
            }
        });

        return [button, content];
    }

    function truncateText(text, maxLength = 100) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }

    function createToggleableText(text) {
        const container = document.createElement('span');
        const shortText = document.createElement('span');
        shortText.textContent = truncateText(text);
        const fullText = document.createElement('span');
        fullText.textContent = text;
        fullText.style.display = 'none';
        fullText.style.whiteSpace = 'pre-wrap';
        fullText.style.wordWrap = 'break-word';

        const toggle = document.createElement('span');
        toggle.className = 'toggle-text text-primary';
        toggle.style.cursor = 'pointer';
        toggle.textContent = ' (show more)';

        toggle.addEventListener('click', function() {
            if (fullText.style.display === 'none') {
                shortText.style.display = 'none';
                fullText.style.display = 'inline';
                this.textContent = ' (show less)';
            } else {
                shortText.style.display = 'inline';
                fullText.style.display = 'none';
                this.textContent = ' (show more)';
            }
        });

        container.appendChild(shortText);
        container.appendChild(fullText);
        container.appendChild(toggle);
        return container;
    }

    function processJsonItem(item) {
        const itemContainer = document.createElement('div');
        itemContainer.className = 'json-item card';

        const title = document.createElement('h2');
        title.className = 'card-header bg-primary text-white';
        title.textContent = item.type || 'Untitled';
        itemContainer.appendChild(title);

        const content = document.createElement('div');
        content.className = 'card-body';

        // if item type is output 
        if (item.type === 'output' | item.type === 'explanation' | item.type === 'summarization') {
            // Display prompt_template first
            if (item.prompt_template) {
                const promptTemplate = document.createElement('div');
                promptTemplate.className = 'mb-4';
                promptTemplate.innerHTML = '<h3>Prompt Template:</h3>';
                promptTemplate.appendChild(createToggleableText(item.prompt_template));
                content.appendChild(promptTemplate);
            }

            // Display other items
            for (const [key, value] of Object.entries(item)) {
                if (key !== 'type' && key !== 'prompt_template') {
                    const keyValuePair = document.createElement('div');
                    keyValuePair.className = 'key-value-pair mb-3';

                    const keyElement = document.createElement('h4');
                    keyElement.textContent = key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ') + ':';
                    keyValuePair.appendChild(keyElement);

                    if (typeof value === 'object' && value !== null) {
                        const [button, collapsibleContent] = createCollapsible(key, value);
                        keyValuePair.appendChild(button);
                        keyValuePair.appendChild(collapsibleContent);
                    } else if (typeof value === 'string' && value.length > 100) {
                        keyValuePair.appendChild(createToggleableText(value));
                    } else {
                        const valueElement = document.createElement('p');
                        valueElement.textContent = value;
                        keyValuePair.appendChild(valueElement);
                    }

                    content.appendChild(keyValuePair);
                }
            }
        }

        // if item type is explanation 


        // if item typ is summarization 





        itemContainer.appendChild(content);
        return itemContainer;
    }

    function createNavigationControls() {
        const controls = document.createElement('div');
        controls.className = 'navigation-controls mb-3';

        const prevButton = document.createElement('button');
        prevButton.textContent = '< Previous';
        prevButton.className = 'btn btn-primary me-2';
        prevButton.addEventListener('click', () => navigateItems(-1));

        const nextButton = document.createElement('button');
        nextButton.textContent = 'Next >';
        nextButton.className = 'btn btn-primary me-2';
        nextButton.addEventListener('click', () => navigateItems(1));

        const indexInput = document.createElement('input');
        indexInput.type = 'number';
        indexInput.min = 1;
        indexInput.max = jsonItems.length;
        indexInput.value = 1;
        indexInput.className = 'form-control d-inline-block me-2';
        indexInput.style.width = '80px';
        indexInput.addEventListener('change', () => {
            const newIndex = parseInt(indexInput.value) - 1;
            if (newIndex >= 0 && newIndex < jsonItems.length) {
                showItem(newIndex);
            }
        });

        const maxIndex = document.createElement('span');
        maxIndex.textContent = `/ ${jsonItems.length}`;

        controls.appendChild(prevButton);
        controls.appendChild(nextButton);
        controls.appendChild(indexInput);
        controls.appendChild(maxIndex);

        return controls;
    }

    function showItem(index) {
        currentIndex = index;
        jsonContainer.innerHTML = '';
        jsonContainer.appendChild(createNavigationControls());
        jsonContainer.appendChild(jsonItems[index]);
        updateNavigationState();
    }

    function navigateItems(direction) {
        const newIndex = currentIndex + direction;
        if (newIndex >= 0 && newIndex < jsonItems.length) {
            showItem(newIndex);
        }
    }

    function updateNavigationState() {
        const indexInput = jsonContainer.querySelector('input[type="number"]');
        indexInput.value = currentIndex + 1;
    }

    const jsonElements = jsonContainer.querySelectorAll('.json-item');
    jsonElements.forEach(item => {
        const parsedItem = JSON.parse(item.querySelector('pre').textContent);
        const processedItem = processJsonItem(parsedItem);
        jsonItems.push(processedItem);
    });

    if (jsonItems.length > 0) {
        showItem(0);
    }
});