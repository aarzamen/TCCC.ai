
To set up and run the FullSubNet integration on the Jetson:

1. First, install dependencies and set up FullSubNet:
   === FullSubNet Setup for TCCC Project ===
This script will download and set up FullSubNet for Nvidia Jetson
Cloning FullSubNet repository...

2. Test the integration with comparison benchmarks:
   

3. Run the microphone-to-text demo with FullSubNet:
   

4. For manual control of enhancement mode:
   Using fullsubnet enhancement mode

5. To use both enhancers in sequence for maximum noise reduction:
   Using both enhancement mode

The FullSubNet enhancer is specifically optimized for the Nvidia Jetson hardware and will use CUDA acceleration automatically when available. If CUDA is not available, it will fall back to CPU processing.

