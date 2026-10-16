# Chartorch

Chartorch is a web-based visual interface that simplifies the creation, training, and testing of machine learning models. By leveraging an intuitive flowchart system, users can design PyTorch neural networks visually without writing boilerplate code, and immediately train or test them on simple datasets. Generated code can be inspected and copied.

Built with Flask, the application is designed with security in mind: 
all user-generated model training and testing code is strictly sandboxed 
and executed within an isolated `nsjail` environment.

## Features

* **Visual Model Builder:** Create PyTorch architectures using a drag-and-drop flowchart interface.
* **Integrated Training & Testing:** Train your custom models on standard datasets directly within the browser.
* **Secure Execution:** Utilizes `nsjail` to safely isolate user-submitted execution processes from the host system.

## Prerequisites

Before installing Chartorch, ensure your system has the following:
* **Linux environment** (Required for `nsjail` process isolation)
* **[nsjail](https://nsjail.dev/)**: Must be installed and accessible on your system path.
* **Python 3.14** (or a compatible Python 3.x version)

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/brandonbate/chartorch.git](https://github.com/brandonbate/chartorch.git)
   cd chartorch

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

### Running the Application
To run the application locally:
   ```bash
   python app.py

Once the server initializes, open your web browser and navigate to: http://localhost:5000
