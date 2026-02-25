// API configuration
const API_BASE_URL = "http://localhost:8000/api";

// DOM elements
const datasetFileInput = document.getElementById("datasetFile");
const datasetSelect = document.getElementById("datasetPath");
const startBtn = document.getElementById("startBtn");
const dataSummary = document.getElementById("dataSummary");
const dataStory = document.getElementById("dataStory");
const htmlTemplateFrame = document.getElementById("htmlTemplateFrame");
const finalStoryFrame = document.getElementById("finalInvFrame");
const selfEvaluateFrame = document.getElementById("selfEvaluateFrame");
const loadingOverlay = document.getElementById("loadingOverlay");
const progressSteps = document.querySelectorAll(".progress-step");

// Event listeners
startBtn.addEventListener("click", startProcess);
datasetFileInput.addEventListener("change", handleFileSelect);

// Handle file selection
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    // Add file selection feedback animation
    const input = event.target;
    input.style.transform = "scale(0.98)";
    setTimeout(() => {
      input.style.transform = "scale(1)";
    }, 150);
  }
}

// Start processing flow
async function startProcess() {
  const datasetPath = datasetFileInput.value.trim() || datasetSelect.value;

  if (!datasetPath) {
    showNotification(
      "Please upload a dataset or select a dataset first",
      "error"
    );
    return;
  }

  showLoading("dataStorySection", true);
  resetProgress();

  // Add button animation
  startBtn.style.transform = "scale(0.95)";
  startBtn.style.boxShadow = "0 2px 10px rgba(52, 152, 219, 0.2)";

  const params = new URLSearchParams({
    dataset_file: "",
    dataset_name: datasetSelect.value,
  });

  // 使用EventSource进行流式接收
  const eventSource = new EventSource(`${API_BASE_URL}/d2inv_stream?${params.toString()}`);
  updateProgress(1, "Processing");

  // 监听后端流式响应
  eventSource.onmessage = function (event) {
    const data = JSON.parse(event.data);

    switch (data.stage) {
      case "data_summary":
//        displayDataSummary({ data_summary: data.data_summary });
        updateProgress(1, "Completed");
        showNotification("Data summary completed", "success");
        updateProgress(2, "Processing");
        break;

      case "data_story":
        showLoading("dataStorySection", false);
        updateProgress(2, "Completed");
        displayDataStory({ data_story: data.data_story });
        showNotification("Data story generated", "success");
        updateProgress(3, "Processing");
        showLoading("htmlTemplateSection", true);
        break;

      case "html_template":
        showLoading("htmlTemplateSection", false);
        updateProgress(3, "Completed");
        displayHtmlTemplate({ html_template: data.html_template });
        showNotification("HTML template generated", "success");
        updateProgress(4, "Processing");
        showLoading("finalInvSection", true);
        break;

      case "inv":
        showLoading("finalInvSection", false);
        updateProgress(4, "Completed");
        // 生成图表
        displayINV({ inv: data.inv })
        showNotification("INV generated", "success");
        updateProgress(5, "Processing");
        showLoading("selfEvaluateSection", true);
        break;

      case "evaluation":
        showLoading("selfEvaluateSection", false);
        displaySelfEvaluation({ self_evaluation: data.self_evaluation })
        updateProgress(5, "Completed");
        showNotification("All processing completed!", "success");
        eventSource.close();
        startBtn.style.transform = "scale(1)";
        startBtn.style.boxShadow = "0 4px 15px rgba(52, 152, 219, 0.3)";
        break;

      default:
        console.log("Unknown stage:", data.stage);
    }
  };

  eventSource.onerror = function (event) {
    console.error("EventSource error:", event);
    eventSource.close();
    showLoading(false);
    startBtn.style.transform = "scale(1)";
    startBtn.style.boxShadow = "0 4px 15px rgba(52, 152, 219, 0.3)";
    showNotification("Connection error during processing", "error");
  };
}

// Display JSON result
function displayDataSummary(result) {
  dataSummary.textContent = JSON.stringify(result.data_summary, null, 4);

  // Add code highlighting effect
  setTimeout(() => {
    dataSummary.style.background =
      "linear-gradient(135deg, #1a2a6c 0%, #2c3e50 50%, #4776E6 100%)";
    setTimeout(() => {
      dataSummary.style.background =
        "linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%)";
    }, 300);
  }, 100);
}

// Display JSON result
function displayDataStory(result) {
  dataStory.textContent = JSON.stringify(result.data_story, null, 4);

  // Add code highlighting effect
  setTimeout(() => {
    dataStory.style.background =
      "linear-gradient(135deg, #1a2a6c 0%, #2c3e50 50%, #4776E6 100%)";
    setTimeout(() => {
      dataStory.style.background =
        "linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%)";
    }, 300);
  }, 100);
}

// Display HTML template
function displayHtmlTemplate(result) {
  htmlTemplateFrame.srcdoc = result.html_template;
}

// Display final story
function displayINV(result) {
  finalStoryFrame.srcdoc = result.inv;
}

// Display self evaluation
function displaySelfEvaluation(result) {
  selfEvaluateFrame.srcdoc = result.self_evaluation;
}

// Update progress
function updateProgress(step, status) {
  progressSteps.forEach((stepEl, index) => {
    if (index + 1 === step) {
      if (status === "Processing") {
        stepEl.classList.add("active");
        stepEl.classList.remove("completed");
        // Add pulse animation
        stepEl.style.animation = "pulse 0.5s ease";
        setTimeout(() => {
          stepEl.style.animation = "";
        }, 500);
      } else if (status === "Completed") {
        stepEl.classList.remove("active");
        stepEl.classList.add("completed");
        // Add completion animation
        stepEl.style.transform = "scale(1.05)";
        setTimeout(() => {
          stepEl.style.transform = "scale(1)";
        }, 300);
      }
    } else if (index + 1 < step) {
      stepEl.classList.remove("active");
      stepEl.classList.add("completed");
    }
  });
}

// Reset progress
function resetProgress() {
  progressSteps.forEach((stepEl) => {
    stepEl.classList.remove("active", "completed");
  });
  progressSteps[0].classList.add("active");
}

// Show/hide loading animation
function showLoading(sectionId, show) {
  const section = document.getElementById(sectionId);
  if (section) {
    const loadingOverlay = section.querySelector('.loading-overlay');
    if (loadingOverlay) {
      if (show) {
        loadingOverlay.classList.add("show");
      } else {
        loadingOverlay.classList.remove("show");
      }
    }
  }
}

// Show notification
function showNotification(message, type) {
  // Create notification element
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;
  notification.textContent = message;

  // Set styles
  Object.assign(notification.style, {
    position: "fixed",
    top: "20px",
    right: "20px",
    padding: "15px 25px",
    borderRadius: "8px",
    color: "white",
    fontWeight: "500",
    zIndex: "10001",
    transform: "translateX(100%)",
    transition: "transform 0.3s ease",
    boxShadow: "0 5px 15px rgba(0,0,0,0.2)",
  });

  if (type === "success") {
    notification.style.background =
      "linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)";
  } else if (type === "error") {
    notification.style.background =
      "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)";
  }

  document.body.appendChild(notification);

  // Show animation
  setTimeout(() => {
    notification.style.transform = "translateX(0)";
  }, 100);

  // Auto remove
  setTimeout(() => {
    notification.style.transform = "translateX(100%)";
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 3000);
}

// Utility function: delay
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 加载datasets目录下的文件列表
async function loadDatasetFiles() {
  try {
    // 从后端API获取datasets目录下的文件列表
    const response = await fetch(`${API_BASE_URL}/list_datasets`);
    const files = await response.json();

    // 清空现有选项
    datasetSelect.innerHTML =
      '<option value="">Select a dataset...</option>';

    // 添加文件选项
    files.forEach((file) => {
      const option = document.createElement("option");
      option.value = file;
      option.textContent = file;
      datasetSelect.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading dataset files:", error);
    showNotification("Error loading dataset files","error")
  }
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  console.log("D2INV Web application loaded");
  resetProgress();

  // Add page loading animation
  document.body.style.opacity = "0";
  document.body.style.transition = "opacity 0.5s ease";
  setTimeout(() => {
    document.body.style.opacity = "1";
  }, 100);
  loadDatasetFiles();
});
