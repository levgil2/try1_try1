import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")





# Load images
image = load_image("sample.png")
#image = load_image("sample_with_hand_write.png")
#image = load_image('drive_licence.png')

# convert image to grayscale
image = image.convert("L")

# convert image to RGB
image = image.convert("RGB")



# Initialize processor and model
processor = AutoProcessor.from_pretrained(r'D:\models\smol_docliing')
model = AutoModelForVision2Seq.from_pretrained(
    r'D:\models\smol_docliing',

    _attn_implementation="eager",  # for gpu that does not supports flash attention
).to(DEVICE)

#Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "This is an article, extract the markdown with bounding boxes."},
        ]
    },
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "just perform OCR,"},
#         ]
#     },
# ]

# start measure time
import time
start = time.time()

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# end measure time and print
end = time.time()
print(f"Time taken to upload inputs to device: {end - start} seconds")

start = time.time()
# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=8192)
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()

end = time.time()
print(f"Time taken to generate: {end - start} seconds")


start = time.time()
# Populate document
doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
print(doctags)
# create a docling document
doc = DoclingDocument(name="Document")
doc.load_from_doctags(doctags_doc)

end = time.time()
print(f"Time taken to Populate document: {end - start} seconds")

doc.save_as_markdown(r"mark_down.json")
print(doc.export_to_markdown())
print(doc.export_to_text())