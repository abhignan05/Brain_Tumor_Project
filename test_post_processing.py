from post_processing import analyze_tumor_region

image_path = "Dataset/Testing/notumor/Te-no_0014.jpg"

result = analyze_tumor_region(image_path)

print("\nResult:", result["result"])
print("Predicted Class:", result["predicted_class"])
print("Confidence:", result["confidence"], "%")
print("Tumor Size:", result["tumor_percentage"], "%")
print("Risk Level:", result["risk_level"])
print("Precaution:", result["precaution"])
print("Recommendation:", result["recommendation"])
