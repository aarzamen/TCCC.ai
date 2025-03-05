# TCCC Casualty Card (DD Form 1380) Implementation

## Overview

The TCCC Casualty Card (DD Form 1380) is a critical battlefield documentation tool used to record essential information about casualties. This module will implement automated completion of this form based on audio transcriptions and LLM analysis.

## Reference Material

- Official form template: https://tccc.org.ua/files/downloads/tccc-cpp-skill-card-55-dd-1380-tccc-casualty-card-en.pdf
- Importance: Equal to 9-line MEDEVAC and ZMIST record capabilities

## Requirements

### Functional Requirements

1. **Audio Processing**
   - Extract casualty information from audio transcriptions
   - Map extracted information to DD Form 1380 fields

2. **Form Generation**
   - Create digital representation of DD Form 1380
   - Support PDF export for documentation
   - Enable review/editing before finalization

3. **Integration Points**
   - Document Library: Store completed forms
   - LLM Analysis: Extract relevant medical information
   - STT Engine: Process audio input for key casualty data
   - Data Store: Persist form data and track completion

### Technical Requirements

1. **Core Components**
   - Form template generator
   - Field extraction pipeline
   - Validation rules engine
   - PDF generation service

2. **Implementation Approach**
   - Field mapping schema for translating audio context to form fields
   - Confidence scoring for extracted information
   - Missing information detection and prompting
   - Hierarchical information extraction prioritizing critical data

## Implementation Plan

1. **Phase 1: Field Schema Development**
   - Define all DD Form 1380 fields and data types
   - Create validation rules for each field
   - Develop information extraction patterns

2. **Phase 2: Form Generation**
   - Create form templates
   - Implement PDF generation
   - Build form preview capability

3. **Phase 3: Integration**
   - Connect with STT Engine for audio input
   - Integrate with LLM Analysis for information extraction
   - Link to Document Library for storage
   - Connect with Data Store for persistence

4. **Phase 4: Testing & Validation**
   - Test with sample audio recordings
   - Validate form completeness and accuracy
   - Stress test with varied casualty scenarios
   - Performance optimization for edge devices

## Success Criteria

1. Accurately extract >90% of casualty information from clear audio input
2. Generate properly formatted DD Form 1380 PDFs
3. Complete form within 60 seconds of audio completion
4. Function effectively on Jetson Orin Nano hardware