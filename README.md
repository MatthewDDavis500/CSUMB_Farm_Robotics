<h1>Amiga Person Follower (Dual OAK-D)</h1>
<br>
<p>This repository contains an autonomous person-following application designed for the farm-ng Amiga robot platform. It utilizes two OAK-D cameras for two-lane monitoring and the YOLOv8 model for real-time person detection. The repository also currently includes farm-ng example code as well as Farah and Hamza's camera-based following code (https://github.com/MUPAAL/Farm-NG-Follow-AI-CAM/tree/main) for reference.</p>

<br><br><br>

<h3>Overview</h3>
<br><br>
<p>The system runs two concurrent camera workers that feed into a central control loop.</p>
<br>
<p>Dual-Camera Functionality: The robot constantly monitors both camera feeds and automatically targets the closest person, determined by the bounding box dimensions.</p>
<br>
<p>Smoothing: To prevent the robot from jittering during small detection fluctuations, an Exponential Moving Average is applied to the target's horizontal center.</p>
<br>
<p>Safety Timeout: If the target is lost, the robot enters a soft stop phase, gradually ramping down speed over a set duration instead of stopping abruptly.</p>

<br><br><br>

<h3>Hardware Requirements</h3>
<br><br>
<p>Robot: farm-ng Amiga</p>
<br>
<p>Sensors: Two OAK-D Cameras (configured as oak0 and oak1 in the oak_config.json file)</p>
<br>
<p>Compute: Amiga Brain</p>

<br><br><br>

<h3>Installation and Setup</h3>
<br>
<ol>
    <li>Clone the repository to the Amiga Brain.</li>
    <li>Install dependencies: pip install -r requirements.txt</li>
    <li>Model Placement: Ensure the YOLO model is located at data/models/yolov8n.pt.</li>
    <li>Configuration: Key parameters can be adjusted directly in the script such as max speeds, target state, etc.</li>
    <li>Run the code from the root directory: python3 src/main.py --ip [YOUR_AMIGA_IP]</li>
</ol>

<br><br><br>

<h3>Controls:</h3>
<br><br>
<p>Visualizer: Three windows will open (oak0, oak1, and ACTIVE_TARGET).</p>
<br>
<p>Quit: Press 'q' while any OpenCV window is active to stop the robot and exit the program.</p>

<br><br><br>

<h3>Safety Note:</h3>
<p>This application is intended for research and development. Autonomous systems can behave unexpectedly in complex environments or due to sensor interference. Always maintain a clear line of sight to the robot and be prepared to engage the physical E-Stop manually.</p>