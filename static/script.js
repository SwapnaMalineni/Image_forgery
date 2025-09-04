document.querySelector('.menu-header').addEventListener('click', function (e) {
  e.preventDefault();
  document.querySelector('.sidebar').classList.toggle('collapsed');
});

window.addEventListener('resize', () => {
  if (window.innerWidth > 768) {
    sidebar.classList.remove('active'); // Reset mobile state
  }
});

document.addEventListener('click', (e) => {
  if (window.innerWidth <= 768 &&
    !sidebar.contains(e.target) &&
    !menuHeader.contains(e.target)) {
    sidebar.classList.remove('active');
  }
});

// Upload and Analysis Section
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const detailedImage = document.getElementById('detailedImage');
const detailedAnalysis = document.getElementById('detailedAnalysis');
const loadingSpinner = document.getElementById('loading');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusText = document.getElementById('statusText');
const confidenceValue = document.getElementById('confidenceValue');
const elaValue = document.getElementById('elaValue');
const noiseValue = document.getElementById('noiseValue');
const metadataValue = document.getElementById('metadataValue');
const metricsContent = document.getElementById('metricsContent');
const skeletonMetrics = document.getElementById('skeletonMetrics');

// Add event listeners to menu items
const menuItems = document.querySelectorAll('.menu-item');
menuItems.forEach(item => {
    item.addEventListener('click', () => {
        // Remove active class from all menu items
        menuItems.forEach(i => i.classList.remove('active'));

        // Add active class to the clicked menu item
        item.classList.add('active');
    });
});

// File input events
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      // Set the same image for detailed analysis
      detailedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

// Analyze Image Function
function analyzeImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading spinner and set status text
    loadingSpinner.style.display = 'flex';
    statusText.textContent = "Image Analysis in Process..."; // Set loading text
    statusText.classList.remove('authentic', 'forged'); // Remove any previous classes

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Hide loading spinner
        loadingSpinner.style.display = 'none';

        // Update the status text below the header
        if (data.class_label) {
            // Set the dynamic text based on the result
            statusText.textContent = data.class_label.toLowerCase() === "authentic" 
                ? "Your image is Authentic" 
                : "Your image is Forged";

            // Add a CSS class based on the class_label
            if (data.class_label.toLowerCase() === "authentic") {
                statusText.classList.remove('forged');  // Remove forged class if present
                statusText.classList.add('authentic');  // Add authentic class
            } else if (data.class_label.toLowerCase() === "forged") {
                statusText.classList.remove('authentic');  // Remove authentic class if present
                statusText.classList.add('forged');  // Add forged class
            }
        } else {
            statusText.textContent = "Image Analysis";  // Fallback status
            statusText.classList.remove('authentic', 'forged');  // Remove both classes
        }

        // Update the detailed analysis section
        const detailedImage = document.getElementById('detailedImage');
        if (data.image) {
            detailedImage.src = `/uploads/${data.image}`;  // Set the image source
            document.getElementById('detailedAnalysis').style.display = 'block';  // Show the detailed analysis section
        } else {
            detailedImage.src = '';  // Clear the image source if no image is available
            document.getElementById('detailedAnalysis').style.display = 'none';  // Hide the detailed analysis section
        }

        // Update other metrics (confidence, ELA variance, etc.)
        document.getElementById('confidenceValue').textContent = data.confidence;
        document.getElementById('elaValue').textContent = data.ela_variance;
        document.getElementById('noiseValue').textContent = data.noise_consistency;
        document.getElementById('metadataValue').innerHTML = data.metadata;  // Render metadata as HTML
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred during analysis.");
    });
}

// Update Results Function
function updateResults(results) {
  statusText.textContent = results.class_label === 'Authentic' ? 'Your image is Authentic' : 'Your image is Forged';
  statusText.className = results.class_label.toLowerCase();
  confidenceValue.textContent = results.confidence;
  elaValue.textContent = results.ela_variance;
  noiseValue.textContent = results.noise_consistency;

  // Render metadata as HTML
  metadataValue.innerHTML = results.metadata || 'N/A';

  // Show the detailed analysis section only when the image is forged
  if (results.class_label === 'Forged') {
    detailedAnalysis.style.display = 'block';
    detailedImage.src = results.image || ''; // Set the highlighted image if available
  } else {
    detailedAnalysis.style.display = 'none';
  }
}

function clearImageAndAnalysis() {
  // Clear the image preview
  const imagePreview = document.getElementById('imagePreview');
  imagePreview.src = ''; // Remove the image source
  imagePreview.style.display = 'none'; // Hide the preview container

  // Clear the file input
  const fileInput = document.getElementById('fileInput');
  fileInput.value = ''; // Reset the file input

  // Clear the analysis results
  const statusText = document.getElementById('statusText');
  statusText.textContent = ''; // Clear the status text

  const confidenceValue = document.getElementById('confidenceValue');
  confidenceValue.textContent = '-'; // Reset confidence value

  const elaValue = document.getElementById('elaValue');
  elaValue.textContent = '-'; // Reset ELA variance value

  const noiseValue = document.getElementById('noiseValue');
  noiseValue.textContent = '-'; // Reset noise consistency value

  const metadataValue = document.getElementById('metadataValue');
  metadataValue.textContent = '-'; // Reset metadata value

  // Hide the detailed analysis section
  const detailedAnalysis = document.getElementById('detailedAnalysis');
  detailedAnalysis.style.display = 'none';

  // Clear any success or error messages
  const successMessage = document.getElementById('successMessage');
  successMessage.textContent = '';
}


// Reset Metrics Function
function resetMetrics() {
  confidenceValue.textContent = '-';
  elaValue.textContent = '-';
  noiseValue.textContent = '-';
  metadataValue.textContent = '-';
}