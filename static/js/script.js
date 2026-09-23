// script.js
// Handles the prediction form submission, renders the result card,
// loads model performance metrics, and draws the two charts on the
// prediction page using Chart.js.

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  if (form) {
    setupPredictForm(form);
  }

  const metricGrid = document.getElementById("metricGrid");
  if (metricGrid) {
    loadMetrics();
  }

  const actualPredictedCanvas = document.getElementById("actualPredictedChart");
  if (actualPredictedCanvas) {
    loadActualVsPredictedChart(actualPredictedCanvas);
  }

  const bmiCanvas = document.getElementById("bmiChart");
  if (bmiCanvas) {
    loadBmiChart(bmiCanvas);
  }
});

function setupPredictForm(form) {
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const submitLabel = document.getElementById("submitLabel");
  const formMsg = document.getElementById("formMsg");
  const placeholder = document.getElementById("resultPlaceholder");
  const resultCard = document.getElementById("resultCard");
  const resultAmount = document.getElementById("resultAmount");
  const summaryList = document.getElementById("summaryList");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    formMsg.style.display = "none";

    const formData = new FormData(form);
    const payload = {
      age: formData.get("age"),
      sex: formData.get("sex"),
      bmi: formData.get("bmi"),
      children: formData.get("children"),
      smoker: formData.get("smoker"),
      region: formData.get("region"),
    };

    setLoading(true);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || !data.success) {
        const errors = data.errors || ["Something went wrong. Please check your inputs."];
        formMsg.textContent = errors.join(" ");
        formMsg.style.display = "block";
        setLoading(false);
        return;
      }

      renderResult(data);
    } catch (err) {
      formMsg.textContent = "Could not reach the prediction server. Please try again.";
      formMsg.style.display = "block";
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    spinner.style.display = isLoading ? "inline-block" : "none";
    submitLabel.textContent = isLoading ? "Predicting..." : "Predict Insurance Cost";
  }

  function renderResult(data) {
    const charge = data.prediction;
    const s = data.input_summary;

    resultAmount.textContent = formatCurrency(charge);

    const smokerBadge = s.smoker === "yes"
      ? '<span class="badge badge-danger">Smoker</span>'
      : '<span class="badge badge-success">Non-smoker</span>';

    summaryList.innerHTML = `
      <div class="quote-slip-row"><span class="k">Age</span><span class="v">${s.age}</span></div>
      <div class="quote-slip-row"><span class="k">Sex</span><span class="v">${capitalize(s.sex)}</span></div>
      <div class="quote-slip-row"><span class="k">BMI</span><span class="v">${s.bmi}</span></div>
      <div class="quote-slip-row"><span class="k">Children</span><span class="v">${s.children}</span></div>
      <div class="quote-slip-row"><span class="k">Smoker</span><span class="v">${smokerBadge}</span></div>
      <div class="quote-slip-row"><span class="k">Region</span><span class="v">${capitalize(s.region)}</span></div>
    `;

    placeholder.style.display = "none";
    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatCurrency(amount) {
  return "₹ " + Number(amount).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const m = await res.json();
    document.getElementById("metricR2").textContent = m.r2_score;
    document.getElementById("metricAdjR2").textContent = m.adjusted_r2_score;
    document.getElementById("metricMSE").textContent = Number(m.mean_squared_error).toLocaleString("en-IN");
    document.getElementById("metricRMSE").textContent = Number(m.root_mean_squared_error).toLocaleString("en-IN");
  } catch (err) {
    console.error("Failed to load metrics", err);
  }
}

async function loadActualVsPredictedChart(canvas) {
  if (typeof Chart === "undefined") {
    console.error("Chart.js failed to load from the CDN.");
    showChartError(canvas);
    return;
  }
  try {
    const res = await fetch("/static/data/actual_vs_predicted.json");
    const points = await res.json();

    const maxVal = Math.max(...points.map((p) => Math.max(p.actual, p.predicted)));

    new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Test predictions",
            data: points.map((p) => ({ x: p.actual, y: p.predicted })),
            backgroundColor: "rgba(79, 70, 229, 0.55)",
            pointRadius: 3.5,
          },
          {
            label: "Perfect prediction",
            type: "line",
            data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
            borderColor: "rgba(139, 92, 246, 0.6)",
            borderWidth: 2,
            borderDash: [6, 6],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: chartOptions("Actual charges", "Predicted charges"),
    });
  } catch (err) {
    console.error("Failed to load actual-vs-predicted chart data", err);
  }
}

async function loadBmiChart(canvas) {
  if (typeof Chart === "undefined") {
    console.error("Chart.js failed to load from the CDN.");
    showChartError(canvas);
    return;
  }
  try {
    const res = await fetch("/static/data/bmi_vs_charges.json");
    const points = await res.json();

    const smokers = points.filter((p) => p.smoker === "yes");
    const nonSmokers = points.filter((p) => p.smoker === "no");

    new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Non-smoker",
            data: nonSmokers.map((p) => ({ x: p.bmi, y: p.charges })),
            backgroundColor: "rgba(79, 70, 229, 0.5)",
            pointRadius: 3,
          },
          {
            label: "Smoker",
            data: smokers.map((p) => ({ x: p.bmi, y: p.charges })),
            backgroundColor: "rgba(224, 69, 95, 0.6)",
            pointRadius: 3,
          },
        ],
      },
      options: chartOptions("BMI", "Charges"),
    });
  } catch (err) {
    console.error("Failed to load bmi chart data", err);
  }
}

function showChartError(canvas) {
  const wrap = canvas.closest(".chart-wrap");
  if (wrap) {
    wrap.innerHTML =
      '<p style="color:var(--text-faint); font-size:0.85rem; padding-top:24px; text-align:center;">' +
      "Chart could not load. Check your internet connection and refresh." +
      "</p>";
  }
}

function chartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: { boxWidth: 10, font: { family: "Inter", size: 11 } },
      },
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, font: { family: "Inter", size: 11 } },
        grid: { color: "#eeecf6" },
      },
      y: {
        title: { display: true, text: yLabel, font: { family: "Inter", size: 11 } },
        grid: { color: "#eeecf6" },
      },
    },
  };
}