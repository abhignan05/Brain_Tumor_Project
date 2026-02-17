from post_processing import analyze_tumor_region

image_path = "Dataset/Testing/meningioma/Te-me_0040.jpg"

result = analyze_tumor_region(image_path)

print("\nResult:", result["result"])
print("Predicted Class:", result["predicted_class"])
print("Confidence:", result["confidence"], "%")

if result["result"] == "Tumor Detected":
    print("Tumor Size:", result["tumor_percentage"], "%")
