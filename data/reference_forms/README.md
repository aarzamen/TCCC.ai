# Reference Forms

This directory contains reference materials for various medical forms that the TCCC.ai system needs to process.

## Forms

### DD Form 1380: TCCC Casualty Card

- Official reference: https://tccc.org.ua/files/downloads/tccc-cpp-skill-card-55-dd-1380-tccc-casualty-card-en.pdf
- Implementation priority: HIGH
- Status: Pending implementation
- Required fields:
  - Casualty information (name, rank, SSN)
  - Injury details and mechanism
  - Treatment provided
  - Vital signs timeline
  - Evacuation details

## Usage

These reference forms should be used by the Document Library module to:

1. Train information extraction models
2. Define form field schemas
3. Test field mapping accuracy
4. Generate digital templates

## Implementation Notes

The system should prioritize capturing critical medical information from audio transcripts and accurately mapping this information to the appropriate form fields.