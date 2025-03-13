document.addEventListener("DOMContentLoaded", function () {
    const imageUpload = document.getElementById("image-upload");
    const imagePreview = document.getElementById("image-preview");
    const imageAnalysisResult = document.getElementById("image-analysis-result");

    // 📸 **Resim Yükleme ve Analiz**
    imageUpload.addEventListener("change", async function () {
        const file = imageUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(file);

        try {
            const response = await fetch("/image-analysis", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            let output = `📊 **Resim Analizi Sonucu**\n`;

            if (data.analysis.yuzler.length) {
                output += `👤 **Yüz:** ${data.analysis.yuzler.join(", ")}\n`;
            }
            if (data.analysis.nesneler.length) {
                output += `🛠️ **Nesneler:** ${data.analysis.nesneler.join(", ")}\n`;
            }
            if (data.analysis.renkler.length) {
                output += `🎨 **Baskın Renkler:** ${data.analysis.renkler.join(", ")}\n`;
            }
            if (data.analysis.metinler.length) {
                output += `📝 **Tespit Edilen Metin:** ${data.analysis.metinler.join(" ")}\n`;
            }

            imageAnalysisResult.textContent = output;
        } catch (error) {
            imageAnalysisResult.textContent = "⚠️ Resim yüklenirken hata oluştu.";
        }
    });
});
