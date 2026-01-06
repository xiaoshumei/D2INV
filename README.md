# D2INV: Automatic Generation of Interactive Narrative Visualizations using Large Language Models for Data Storytelling

D2INV is a library for generating interactive narrative visualizations and data stories using large language models. It
supports multiple dataset formats and provides an automated pipeline for summarization, storytelling, and visualization.

## Features

- Data Summarization
- Automatic Data Story Generation
- Visualization Template Generation
- Automatic Data Visualization
- Evaluation of Generated INVs

## Main Pipeline Overview

The main execution flow in `api/app.py` consists of the following steps:

1. **Environment Setup**  
   Loads environment variables and configuration from `.env` to initialize API keys and parameters.

2. **Result Directory Preparation**  
   For each dataset, creates a dedicated results folder to store outputs.

3. **Data Summarization**  
   Loads the dataset and generates a summary using statistical analysis and feature extraction.

4. **Summary Output**  
   Writes the generated summary to the results directory for reference and downstream tasks.

5. **Data Story Generation**  
   Uses large language models to generate a narrative data story based on the dataset and its summary.

6. **Visualization Template Generation**  
   Intended to generate visualization templates for the data story.

7. **Data Visualization Generation**
   Generates visualizations based on the data story and templates.

8. **INV Generation**
   Combines the visualization template and data visualizations to create interactive narrative visualizations.

9. **Self-Evaluation**
   Automatically evaluates the quality of generated stories and visualizations using LLM-based assessment.

## Example Usage

1. Put the dataset in the `datasets` directory and run the following command:
2. Clone the repository
3. Set LLM_API_KEY in .env
4. Run the following command:

```sh
git clone https://github.com/xiaoshumei/D2INV.git
cd D2INV
python api/app.py --dataset datasets/example.csv
```

## Project Structure

```
.env
requirements.txt
api/
datasets/
results/
tokenizer/
tools/
```

## Installation

```sh
pip install -r requirements.txt
```

## Start a local server to preview the generated INVs

```sh
python api_server.py
```

Then open http://localhost:8000 in your browser.