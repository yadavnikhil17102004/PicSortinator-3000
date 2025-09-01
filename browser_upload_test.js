// 🧪 Browser Console Test for Upload Functionality
// Open your browser, go to http://localhost:5000/upload
// Open browser console (F12), paste this code and run it

console.log("🧪 Testing PicSortinator 3000 upload...");

// Create a test blob (fake image)
const canvas = document.createElement('canvas');
canvas.width = 100;
canvas.height = 100;
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'red';
ctx.fillRect(0, 0, 100, 100);

canvas.toBlob(function(blob) {
    const formData = new FormData();
    formData.append('files[]', blob, 'test.png');
    
    console.log("📤 Sending test upload...");
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("📊 Response status:", response.status);
        return response.json();
    })
    .then(data => {
        console.log("✅ Upload response:", data);
        if (data.success) {
            console.log("🎉 Upload test successful!");
        } else {
            console.log("❌ Upload test failed:", data.message);
        }
    })
    .catch(error => {
        console.log("💥 Upload error:", error);
    });
}, 'image/png');

console.log("🕐 Test started... check output above in a moment!");
