// Model selector component for integration with your UI

class ModelSelectorUI {
  constructor(processor, containerElement) {
    this.processor = processor;
    this.container = containerElement;
    this.models = [];
    this.selectedModelId = null;
    this.initialized = false;
    this.onModelChange = null; // Callback for when model changes
  }
  
  /**
   * Initialize the model selector
   * @param {Function} onModelChange - Optional callback when model changes
   */
  async initialize(onModelChange = null) {
    if (this.initialized || !this.container) return;
    
    this.onModelChange = onModelChange;
    
    try {
      // Fetch available models
      await this.refreshModels();
      
      // Create model selector UI
      this.container.innerHTML = `
        <div class="model-selector">
          <div class="selector-header">
            <h3>Model Selection</h3>
            <button class="refresh-button">↻</button>
          </div>
          
          <div class="models-container">
            ${this.renderModelOptions()}
          </div>
          
          <div class="model-details">
            ${this.renderModelDetails()}
          </div>
        </div>
      `;
      
      // Add event listeners
      this.attachEventListeners();
      
      this.initialized = true;
    } catch (error) {
      console.error('Failed to initialize model selector:', error);
      this.container.innerHTML = `
        <div class="error-message">
          Failed to load models: ${error.message}
          <button class="retry-button">Retry</button>
        </div>
      `;
      
      // Add retry button listener
      const retryButton = this.container.querySelector('.retry-button');
      if (retryButton) {
        retryButton.addEventListener('click', () => this.initialize(onModelChange));
      }
    }
  }
  
  /**
   * Fetch available models from the API
   */
  async refreshModels() {
    const data = await this.processor.getAvailableModels();
    this.models = data.models;
    this.selectedModelId = data.current_model.id;
  }
  
  /**
   * Render the model selection options
   * @returns {string} HTML for model options
   */
  renderModelOptions() {
    if (!this.models || this.models.length === 0) {
      return '<p>No models available</p>';
    }
    
    return `
      <select class="model-select">
        ${this.models.map(model => `
          <option value="${model.id}" ${model.id === this.selectedModelId ? 'selected' : ''}>
            ${model.name}
          </option>
        `).join('')}
      </select>
    `;
  }
  
  /**
   * Render details for the currently selected model
   * @returns {string} HTML for model details
   */
  renderModelDetails() {
    if (!this.selectedModelId) return '<p>No model selected</p>';
    
    const model = this.models.find(m => m.id === this.selectedModelId);
    if (!model) return '<p>Model not found</p>';
    
    return `
      <div class="model-info">
        <h4>${model.name}</h4>
        <p>${model.description || 'No description available'}</p>
        
        <div class="model-params">
          <div class="param-item">
            <span class="param-label">Type:</span>
            <span class="param-value">${model.type}</span>
          </div>
          
          <div class="param-item">
            <span class="param-label">Script:</span>
            <span class="param-value">${model.script_path}</span>
          </div>
          
          ${model.prompt ? `
            <div class="param-item">
              <span class="param-label">Prompt:</span>
              <span class="param-value">${model.prompt}</span>
            </div>
          ` : ''}
          
          <div class="param-item">
            <span class="param-label">Checkpoint:</span>
            <span class="param-value">${model.checkpoint_path}</span>
          </div>
          
          ${Object.keys(model.params || {}).length > 0 ? `
            <div class="param-item">
              <span class="param-label">Additional Parameters:</span>
              <ul class="params-list">
                ${Object.entries(model.params).map(([key, value]) => `
                  <li><span class="param-name">${key}:</span> ${value}</li>
                `).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }
  
  /**
   * Attach event listeners to UI elements
   */
  attachEventListeners() {
    // Model selection change
    const selectElement = this.container.querySelector('.model-select');
    if (selectElement) {
      selectElement.addEventListener('change', async (event) => {
        const modelId = event.target.value;
        
        try {
          // Show loading state
          this.container.querySelector('.model-details').innerHTML = `
            <div class="loading">Loading model details...</div>
          `;
          
          // Select the model
          await this.processor.selectModel(modelId);
          this.selectedModelId = modelId;
          
          // Update details
          this.container.querySelector('.model-details').innerHTML = this.renderModelDetails();
          
          // Call change callback if provided
          if (this.onModelChange) {
            const selectedModel = this.models.find(m => m.id === modelId);
            this.onModelChange(selectedModel);
          }
        } catch (error) {
          console.error('Error selecting model:', error);
          this.container.querySelector('.model-details').innerHTML = `
            <div class="error-message">
              Failed to select model: ${error.message}
            </div>
          `;
        }
      });
    }
    
    // Refresh button
    const refreshButton = this.container.querySelector('.refresh-button');
    if (refreshButton) {
      refreshButton.addEventListener('click', async () => {
        try {
          // Show loading state
          refreshButton.disabled = true;
          refreshButton.textContent = '...';
          
          // Refresh models
          await this.refreshModels();
          
          // Update UI
          this.container.querySelector('.models-container').innerHTML = this.renderModelOptions();
          this.container.querySelector('.model-details').innerHTML = this.renderModelDetails();
          
          // Re-attach event listeners
          this.attachEventListeners();
        } catch (error) {
          console.error('Error refreshing models:', error);
          this.container.querySelector('.model-details').innerHTML = `
            <div class="error-message">
              Failed to refresh models: ${error.message}
            </div>
          `;
        } finally {
          refreshButton.disabled = false;
          refreshButton.textContent = '↻';
        }
      });
    }
  }
}

// Export the component
export default ModelSelectorUI;