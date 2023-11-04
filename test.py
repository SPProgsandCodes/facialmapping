from fastapi import FastAPI, UploadFile, File
import face_recognition as fc
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from typing import Optional, Tuple

app = FastAPI()

@app.post("/recognize-face/")
async def recognize_face(image: UploadFile):
    try:
        # Load the new face image
        
        new_face_image = fc.load_image_file(image.file)
        new_face_encodings = fc.face_encodings(new_face_image)

        if not new_face_encodings:
            return {"message": "No face found in the new image."}

        new_face_encoding = new_face_encodings[0]

        # Load the known face images and compute their encodings
        known_face_encodings = []
        known_face_paths = [
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\subham.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\aman.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_4.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_5.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_6.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_7.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_9.jpg",
        "C:\\Users\\solan\\OneDrive\\Desktop\\DATABASE\\image_1.jpg"
    ]

        known_data = [["rajput subham","age : 19"],["aman singh","age  :14"],[],["magan","age : 45"],["rajesh","age  : 13"],["purshotam","age : 67"],["kuldeep","age : 23"],["rahul","age : 12"],["rahul","age : 23"]]

        max_similarity = 0
        max_similarity_index = -1

        # Process the uploaded image
        input_image = fc.load_image_file(image.file)
        face_locations = fc.face_locations(input_image)
        input_face_encodings = fc.face_encodings(input_image, face_locations)

        if not input_face_encodings:
            return {"message": "No face found in the uploaded image."}

        input_face_encoding = input_face_encodings[0]

        for i, face_path in enumerate(known_face_paths):
            face_image = fc.load_image_file(face_path)
            face_locations = fc.face_locations(face_image)
            known_face_encodings_for_image = fc.face_encodings(face_image, face_locations)

            if known_face_encodings_for_image:
                known_face_encoding = known_face_encodings_for_image[0]

                similarity = cosine_similarity([new_face_encoding], [known_face_encoding])[0][0]
                difference = 1 - similarity

                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_index = i

        if max_similarity_index != -1:
            known_face_path = known_face_paths[max_similarity_index]

            known_face_image = cv2.imread(known_face_path)
            known_face_images = cv2.resize(known_face_image, (300, 400))
            cv2.imshow("Most Similar Known Face", known_face_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "max_similarity": max_similarity,
            "known_data": known_data[max_similarity_index],
            "Euclidian difference"  :difference
        }
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,  port=8000)
