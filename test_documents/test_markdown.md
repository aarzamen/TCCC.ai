# Test Markdown Document

This is a test markdown document that can be used to verify the RAG document processing capabilities.

## Section 1: Medical Terminology

This section contains some medical terminology that should be recognized by the system:

- Hemorrhage control is a critical part of TCCC
- Apply a tourniquet to stop severe bleeding
- Check for tension pneumothorax in chest injuries
- Ensure proper airway management
- Monitor for signs of shock

## Section 2: Formatting Test

Different markdown formatting:

*Italic text* 
**Bold text**
***Bold and italic***

1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3

> This is a blockquote with important information
> about treatment protocols.

## Section 3: Code Example

```python
def check_vital_signs(patient):
    """Check the vital signs of a patient and return status."""
    if patient.pulse < 60:
        return "Bradycardia detected"
    elif patient.pulse > 100:
        return "Tachycardia detected"
    else:
        return "Normal heart rate"
```

## Conclusion

This test document should be properly processed and indexed in the RAG database.