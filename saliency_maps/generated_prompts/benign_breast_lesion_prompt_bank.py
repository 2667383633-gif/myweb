# Auto-generated lesion-centric prompt bank for subtype=benign

benign_breast_lesion_prompt_bank = {
  "bank_name": "benign_breast_lesion_prompt_bank",
  "subtype": "benign",
  "modality": "breast_ultrasound",
  "prompts": [
    {
      "id": "benign_lesion_01",
      "text": "A breast ultrasound image showing a hypoechoic mass with well-defined margins.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast mass",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "benign_lesion_02",
      "text": "An ultrasound image of the breast demonstrating a focal lesion with posterior acoustic enhancement.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "benign_lesion_03",
      "text": "A breast ultrasound image depicting an oval-shaped mass with smooth margins.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast mass",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "benign_lesion_04",
      "text": "An ultrasound image of the breast showing a circumscribed hypoechoic lesion.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "benign_subtype_01",
      "text": "A breast ultrasound image demonstrating a benign breast lesion with circumscribed margins and posterior enhancement.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": "circumscribed margins, posterior enhancement, posterior shadowing",
        "location": null
      }
    },
    {
      "id": "benign_subtype_02",
      "text": "An ultrasound image of the breast showing a benign breast mass with smooth margins and parallel orientation.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast mass",
        "morphology": "smooth margins, parallel orientation",
        "location": null
      }
    },
    {
      "id": "benign_subtype_03",
      "text": "A breast ultrasound image depicting a benign tumor region with an oval shape and well-defined borders.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign tumor region",
        "morphology": "oval shape, well-defined margins",
        "location": null
      }
    },
    {
      "id": "benign_subtype_04",
      "text": "An ultrasound image of the breast demonstrating a benign breast lesion with round morphology and circumscribed margins.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": "round shape, circumscribed margins",
        "location": null
      }
    },
    {
      "id": "benign_morph_01",
      "text": "A breast ultrasound image showing an oval-shaped mass with smooth margins and posterior acoustic enhancement.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast mass",
        "morphology": "oval shape, smooth margins, posterior enhancement, posterior shadowing",
        "location": null
      }
    },
    {
      "id": "benign_morph_02",
      "text": "An ultrasound image of the breast demonstrating a round lesion with circumscribed margins and parallel orientation.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": "round shape, circumscribed margins, parallel orientation",
        "location": null
      }
    },
    {
      "id": "benign_morph_03",
      "text": "A breast ultrasound image depicting a mass with smooth margins, oval shape, and posterior enhancement.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast mass",
        "morphology": "oval shape, smooth margins, posterior enhancement, posterior shadowing",
        "location": null
      }
    },
    {
      "id": "benign_morph_04",
      "text": "An ultrasound image of the breast showing a circumscribed lesion with round morphology and smooth borders.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "benign breast lesion",
        "morphology": "round shape, circumscribed margins, smooth margins",
        "location": null
      }
    }
  ]
}

# Convenience accessors
benign_breast_lesion_prompt_bank_texts = [p['text'] for p in benign_breast_lesion_prompt_bank['prompts']]
benign_breast_lesion_prompt_bank_pathology_phrases = [p['slots']['pathology'] for p in benign_breast_lesion_prompt_bank['prompts'] if p['slots']['pathology']]
benign_breast_lesion_prompt_bank_morphology_phrases = [p['slots']['morphology'] for p in benign_breast_lesion_prompt_bank['prompts'] if p['slots']['morphology']]
