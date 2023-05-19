from flask import Flask, render_template, request
import os
import shutil

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/browse", methods=["POST"])
def browse():
    file = request.files["file"]
    file.save("selected_file.png")

    destination_dir = r"C:\Users\d\Desktop\University\Graduation project\archive\boneage-test-dataset\boneage-test-dataset"
    for filename in os.listdir(destination_dir):
        file_path = os.path.join(destination_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            return "Error: " + str(e)

    try:
        shutil.copy("selected_file.png", destination_dir)
    except Exception as e:
        return "Error: " + str(e)


    destination_dirc = r"C:\Users\d\Desktop\University\Graduation project\archive\static\result"
    for filename in os.listdir(destination_dirc):
        file_path = os.path.join(destination_dirc, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            return "Error: " + str(e)

    try:
        shutil.copy("selected_file.png", destination_dirc)
    except Exception as e:
        return "Error: " + str(e)
    os.chdir(r"C:\Users\d\Desktop\University\Graduation project\archive")
    os.system("python main.py")

    with open("predicted_months.txt", "r") as file:
        predicted_months = file.read().strip()

    return render_template("results.html", predicted_months=predicted_months)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
    import webbrowser

    webbrowser.open("http://localhost:5000/")
