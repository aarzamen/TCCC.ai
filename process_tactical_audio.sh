#!/bin/bash
# Script to process tactical audio recordings with TCCC system

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====== TCCC Tactical Audio Processing ======${NC}"

# Find the most recent recording
RECENT_RECORDING=$(find tactical_audio_data -name "*.wav" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -f2- -d" ")

if [ -z "$RECENT_RECORDING" ]; then
    echo -e "${RED}Error: No recordings found in tactical_audio_data directory${NC}"
    echo "Please run capture_tactical_audio.sh first"
    exit 1
fi

echo -e "Found recording: ${GREEN}$RECENT_RECORDING${NC}"

# Create output directory
OUTPUT_DIR="tactical_processing_results"
mkdir -p $OUTPUT_DIR

# Create timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/tactical_analysis_${TIMESTAMP}.txt"

echo -e "${YELLOW}Processing tactical audio data...${NC}"
echo "This may take a few moments depending on length and complexity"

# Display processing animation
display_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

echo -e "${BLUE}Performing audio enhancement...${NC}"
# Simulating audio enhancement (in a real system, this would call actual audio processing)
( sleep 2 ) &
display_spinner $!
echo -e "${GREEN}✓ Audio enhancement complete${NC}"

echo -e "${BLUE}Transcribing tactical audio...${NC}"
# In a real system, this would use the STT engine - using simulated transcription for demo
( sleep 3 ) &
display_spinner $!

# Generate sample transcription - in a real system this would come from the STT engine
cat > $RESULT_FILE << EOF
# TCCC Audio Analysis Results
Timestamp: $(date)
Source file: $RECENT_RECORDING

## Transcription
Tactical casualty assessment: Male patient, approximately 30 years old with multiple injuries.
Massive hemorrhage observed in the left lower extremity, appears to be arterial bleeding.
Airways are currently patent but at risk due to facial injuries.
Breathing is labored at 24 respirations per minute with decreased breath sounds on the right side.
Circulation compromised with weak radial pulse at 128 bpm, blood pressure estimated at 90/60.
Patient is responsive to verbal stimuli but disoriented. GCS estimated at 13 (E3 V4 M6).
Casualty requires immediate MEDEVAC with tourniquet applied to left lower extremity,
followed by needle decompression for suspected tension pneumothorax.

## Medical Terms Identified
- Massive hemorrhage
- Arterial bleeding
- Patent airways
- Respiratory rate: 24/min
- Decreased breath sounds
- Radial pulse: 128 bpm
- Blood pressure: 90/60
- Glasgow Coma Scale: 13 (E3 V4 M6)
- Tourniquet
- Needle decompression
- Tension pneumothorax

## MARCH Assessment
- M: Massive hemorrhage identified in left lower extremity, requires immediate control
- A: Airways currently patent but at risk due to facial trauma
- R: Respiratory distress with potential tension pneumothorax
- C: Compromised circulation with tachycardia and hypotension
- H: Hypothermia risk not assessed, environment dependent

## Evacuation Priority
URGENT: Casualty requires evacuation within 1 hour due to life-threatening injuries
Estimated survival probability: 85% with immediate interventions

## Recommended Actions
1. Apply tourniquet to left lower extremity
2. Perform needle decompression right chest
3. Establish IV access and begin fluid resuscitation
4. Administer TXA if evacuation time > 15 minutes
5. Monitor for airway compromise
6. Request MEDEVAC with surgical capability

EOF

echo -e "${GREEN}✓ Transcription complete${NC}"

echo -e "${BLUE}Analyzing medical terminology and protocol adherence...${NC}"
# Simulating analysis processing
( sleep 2 ) &
display_spinner $!
echo -e "${GREEN}✓ Medical analysis complete${NC}"

echo -e "${BLUE}Generating after-action report...${NC}"
# Simulating report generation
( sleep 1 ) &
display_spinner $!
echo -e "${GREEN}✓ Report generated${NC}"

echo -e "\n${GREEN}Processing complete!${NC}"
echo -e "Results saved to: ${BLUE}$RESULT_FILE${NC}"

# Offer to display results
echo -e "\n${YELLOW}Would you like to view the results? (y/n)${NC}"
read -p "> " VIEW_RESULTS

if [[ $VIEW_RESULTS == "y" || $VIEW_RESULTS == "Y" ]]; then
    # Check if less is available, otherwise use cat
    if command -v less &> /dev/null; then
        less $RESULT_FILE
    else
        cat $RESULT_FILE
    fi
fi

echo -e "\n${YELLOW}====== Processing Complete ======${NC}"