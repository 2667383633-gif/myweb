# Auto-generated lesion-centric prompt bank for subtype=malignant

malignant_breast_lesion_prompt_bank = {
  "bank_name": "malignant_breast_lesion_prompt_bank",
  "subtype": "malignant",
  "modality": "breast_ultrasound",
  "prompts": [
    {
      "id": "malignant_lesion_01",
      "text": "A breast ultrasound image showing a hypoechoic mass with indistinct margins.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast mass",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "malignant_lesion_02",
      "text": "An ultrasound image of the breast demonstrating a focal lesion with posterior acoustic shadowing.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast lesion",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "malignant_lesion_03",
      "text": "A breast ultrasound image depicting a solid mass with heterogeneous echotexture.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast mass",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "malignant_lesion_04",
      "text": "A breast ultrasound image showing a malignant hypoechoic focal lesion.",
      "type": "lesion_generic",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast lesion",
        "morphology": null,
        "location": null
      }
    },
    {
      "id": "malignant_subtype_01",
      "text": "A breast ultrasound image demonstrating a malignant breast lesion with irregular shape.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast lesion",
        "morphology": "irregular shape",
        "location": null
      }
    },
    {
      "id": "malignant_subtype_02",
      "text": "An ultrasound image of the breast showing a malignant tumor region with spiculated margins.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant tumor region",
        "morphology": "spiculated margins",
        "location": null
      }
    },
    {
      "id": "malignant_subtype_03",
      "text": "A breast ultrasound image depicting a malignant breast mass with angular margins.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast mass",
        "morphology": "angular margins",
        "location": null
      }
    },
    {
      "id": "malignant_subtype_04",
      "text": "An ultrasound image of the breast demonstrating a malignant lesion with non-parallel orientation.",
      "type": "subtype_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast lesion",
        "morphology": "parallel orientation, non-parallel orientation",
        "location": null
      }
    },
    {
      "id": "malignant_morph_01",
      "text": "A breast ultrasound image showing a hypoechoic mass with irregular shape and spiculated margins.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast mass",
        "morphology": "irregular shape, spiculated margins",
        "location": null
      }
    },
    {
      "id": "malignant_morph_02",
      "text": "An ultrasound image of the breast demonstrating a solid lesion with angular margins and posterior shadowing.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast lesion",
        "morphology": "posterior enhancement, angular margins, posterior shadowing",
        "location": null
      }
    },
    {
      "id": "malignant_morph_03",
      "text": "A breast ultrasound image depicting a heterogeneous mass with non-parallel orientation and irregular margins.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant breast mass",
        "morphology": "parallel orientation, irregular shape, non-parallel orientation",
        "location": null
      }
    },
    {
      "id": "malignant_morph_04",
      "text": "An ultrasound image of the breast showing a hypoechoic tumor region with spiculated margins and posterior acoustic shadowing.",
      "type": "morphology_aware",
      "slots": {
        "modality": "breast ultrasound",
        "anatomy": null,
        "pathology": "malignant tumor region",
        "morphology": "posterior enhancement, spiculated margins, posterior shadowing",
        "location": null
      }
    }
  ]
}

# Convenience accessors
malignant_breast_lesion_prompt_bank_texts = [p['text'] for p in malignant_breast_lesion_prompt_bank['prompts']]
malignant_breast_lesion_prompt_bank_pathology_phrases = [p['slots']['pathology'] for p in malignant_breast_lesion_prompt_bank['prompts'] if p['slots']['pathology']]
malignant_breast_lesion_prompt_bank_morphology_phrases = [p['slots']['morphology'] for p in malignant_breast_lesion_prompt_bank['prompts'] if p['slots']['morphology']]
