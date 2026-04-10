function goChoice() {
  window.location.href = "choice.html";
}

function goAbout() {
  window.location.href = "about.html";
}

function goHome() {
  window.location.href = "index.html";
}

function goNext() {
  window.location.href = "upload.html";
}

function submitImage() {
  const imgInput = document.getElementById("image");

  if (!imgInput || imgInput.files.length === 0) {
    alert("Please upload MRI image");
    return;
  }

  const file = imgInput.files[0];
  const reader = new FileReader();

  reader.onload = function () {
    localStorage.setItem("uploadedMRI", reader.result);
    window.location.href = "result.html";
  };

  reader.readAsDataURL(file);
}


if (window.location.pathname.includes("result.html")) {
  const box = document.getElementById("result");
  const needle = document.getElementById("needle");
  

  const detected = Math.random() > 0.5;

  if (!detected) {
    box.innerHTML = `
      <h3 style="color:#2e7d32;">No Tumour Detected</h3>
      <p><b>Food & Lifestyle Suggestions:</b></p>
      <ul>
        <li>Green leafy vegetables</li>
        <li>Fresh fruits and nuts</li>
        <li>Omega‑3 rich foods</li>
        <li>Daily exercise</li>
        <li>Quality sleep</li>
      </ul>
    `;
    needle.style.transform = "rotate(-45deg)";
  } else {
    box.innerHTML = `
      <h3 style="color:#c62828;">Tumour Detected</h3>
      <p><b>Type:</b> Glioma</p>
      <p><b>Region:</b> Frontal Lobe</p>
      <p><b>Size:</b> 2.5 cm</p>
      <p><b>Risk Stage:</b> Medium</p>
      <p><b>Diet & Medical Advice:</b></p>
      <ul>
        <li>High‑protein balanced diet</li>
        <li>Antioxidant‑rich foods</li>
        <li>Avoid processed food</li>
        <li>Consult neurologist immediately</li>
      </ul>
    `;
    needle.style.transform = "rotate(0deg)";
  }
}
function downloadReport() {
  window.print();
}
