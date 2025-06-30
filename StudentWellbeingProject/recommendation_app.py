# -*- coding: utf-8 -*-
"""
recommendation_app.py -Flask Backend & UI
"""

# Only import what is needed at the top level
def run_flask():
    from flask import Flask
    app = create_app_and_model()
    app.run(debug=True, port=5000, use_reloader=False)

def run_fake_scraper():
    import sys
    import subprocess
    subprocess.run([sys.executable, 'real_twitter_scraper_disabled.py'])

def create_app_and_model():
    import pandas as pd
    import joblib
    from flask import Flask, request, jsonify, render_template_string
    import json
    from pathlib import Path
    app = Flask(__name__)

    def load_latest_model():
        try:
            model_dir = Path.home() / "StudentWellbeingProjectModels"
            performance_log_path = model_dir / 'model_performance.json'
            with open(performance_log_path, 'r') as f:
                performance_data = json.load(f)
            print(f"Loading latest model (Version {performance_data.get('version', 'N/A')}) from:")
            print(f" -> Model: {performance_data['model_file']}")
            model = joblib.load(performance_data['model_file'])
            scaler = joblib.load(performance_data['scaler_file'])
            model_columns = joblib.load(performance_data['columns_file'])
            with open(performance_data['importance_file'], 'r') as f:
                feature_importances = json.load(f)
            return model, scaler, model_columns, feature_importances
        except FileNotFoundError:
            print(f"\n--- WARNING: 'model_performance.json' not found in {Path.home() / 'StudentWellbeingProjectModels'}. ---")
            print("The application will run, but predictions will fail.")
            print("Please run `training_pipeline.py` and `log_updater.py` first.\n")
            return None, None, None, None
        except Exception as e:
            print(f"An error occurred while loading model artifacts: {e}")
            return None, None, None, None

    model, scaler, model_columns, feature_importances = load_latest_model()

    @app.route('/')
    def home():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/analyze-lifestyle', methods=['POST'])
    def analyze_lifestyle():
        if not model: return jsonify({'error': 'Model not available.'})
        data = request.get_json()
        base_inputs = data['base_inputs']
        base_df = pd.DataFrame([base_inputs], columns=model_columns).fillna(0)
        base_scaled = scaler.transform(base_df)
        base_prediction = model.predict(base_scaled)[0]
        current_outcome = {
            'x': float(base_prediction[1]),
            'y': float(base_prediction[0])
        }
        scenarios = {}
        variable_ranges = {
            'studytime': [1, 2, 3, 4],
            'goout': [1, 2, 3, 4, 5],
            'total_alcohol': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        for key, value_range in variable_ranges.items():
            scenario_results = []
            for value in value_range:
                scenario_input = base_inputs.copy()
                scenario_input[key] = value
                input_df = pd.DataFrame([scenario_input], columns=model_columns).fillna(0)
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                scenario_results.append({
                    'x': float(prediction[1]),
                    'y': float(prediction[0]),
                    'value': value
                })
            scenarios[key] = scenario_results
        recommendations = []
        study_time = base_inputs.get('studytime', 2)
        go_out_time = base_inputs.get('goout', 3)
        alcohol_intake = base_inputs.get('total_alcohol', 4)
        social_sentiment = base_inputs.get('social_sentiment', None)
        if alcohol_intake >= 7:
            recommendations.append(f"Your reported alcohol intake is {alcohol_intake} units/week, which is very high. Long-term, heavy consumption is strongly linked to severe health issues, including liver disease, heart problems, and a significantly lower life expectancy. Please consider reducing your intake.")
        elif alcohol_intake >= 5:
            recommendations.append(f"Your alcohol intake is {alcohol_intake} units/week. While it may seem manageable now, consistently consuming this much can negatively impact sleep quality, mental health, and long-term physical health. Try to moderate your drinking habits.")
        if go_out_time <= 2:
            recommendations.append(f"You reported a socializing (go out) score of {go_out_time} (on a 1-5 scale). Prolonged periods of low social interaction can correlate with higher risks of depression and anxiety. Consider scheduling regular, low-stress social activities to protect your well-being.")
        elif go_out_time >= 4:
            recommendations.append(f"Your socializing score is {go_out_time}, which is high. Maintaining strong social connections is great for mental health, but be mindful to balance social time with rest and study.")
        else:
            recommendations.append(f"Your socializing score is {go_out_time}, which is in a healthy range. Keep maintaining your social connections!")
        if study_time >= 3 and go_out_time <= 2:
            recommendations.append(f"You are dedicating a lot of time to studying (score: {study_time}) but have low social activity (score: {go_out_time}). To avoid burnout, schedule restorative breaks and dedicate at least 3-5 hours per week to non-academic hobbies or social time.")
        elif study_time >= 4:
            recommendations.append(f"You are studying more than 10 hours a week (score: {study_time}), which is highly commendable. Ensure you are getting adequate sleep (7-9 hours) and scheduling downtime. A short break every hour of study can also improve focus and retention.")
        else:
            recommendations.append(f"Your study time is {study_time}, which suggests a sustainable balance between your studies and social life. Maintaining this equilibrium is key to long-term success and happiness.")
        if social_sentiment is not None:
            if social_sentiment < -0.5:
                recommendations.append(f"Your recent social sentiment score is {social_sentiment:.2f}, which is quite negative. If you've noticed negative feelings in your social life, consider reaching out to friends, family, or a counselor for support. Engaging in positive social activities may help improve your outlook.")
            elif social_sentiment < -0.1:
                recommendations.append(f"Your social sentiment score is {social_sentiment:.2f}, indicating some negative trends. Try to identify sources of stress or negativity in your social environment and take small steps to address them.")
            elif social_sentiment > 0.5:
                recommendations.append(f"Your social sentiment score is {social_sentiment:.2f}, which is very positive! Keep nurturing these positive connections and habits—they are great for your long-term well-being.")
            elif social_sentiment > 0.1:
                recommendations.append(f"Your social sentiment score is {social_sentiment:.2f}, showing a generally positive outlook. Continue to maintain healthy social interactions and support networks.")
            else:
                recommendations.append(f"Your social sentiment score is {social_sentiment:.2f}, which is neutral. If you feel you need more support, don't hesitate to connect with others or try new social activities.")
        return jsonify({
            'current_outcome': current_outcome,
            'scenarios': scenarios,
            'recommendations': recommendations
        })

    @app.route('/spark-demo')
    def spark_demo():
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import avg
            spark = SparkSession.builder.master('local[*]').appName('Demo').getOrCreate()
            data = [
                ('Alice', 20),
                ('Bob', 23),
                ('Charlie', 22),
                ('Alice', 25)
            ]
            columns = ['name', 'score']
            df = spark.createDataFrame(data, columns)
            result = df.groupBy('name').agg(avg('score').alias('avg_score')).toPandas().to_dict(orient='records')
            spark.stop()
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': str(e)})

    return app

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Well-being Scenario Planner</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        body { font-family: 'Poppins', sans-serif; background-color: #f0f2f5; color: #333; margin: 0; padding: 20px; }
        .main-container { max-width: 900px; margin: auto; background: #fff; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); padding: 40px; }
        h1, h3 { color: #2c3e50; text-align: center; }
        .content-grid { display: grid; grid-template-columns: 300px 1fr; gap: 40px; align-items: start; }
        .controls { display: flex; flex-direction: column; gap: 20px; border-right: 1px solid #ddd; padding-right: 30px; }
        .slider-group label { font-weight: 600; margin-bottom: 10px; display: flex; justify-content: space-between; }
        .slider-group span { font-weight: 400; color: #007bff; }
        input[type="range"] { width: 100%; }
        .chart-container { position: relative; height: 400px; width: 100%; }
        #recommendation-panel { margin-top: 30px; padding: 20px; background: #e9f5ff; border-left: 5px solid #007bff; }
        #recommendation-panel h3 { text-align: left; }
        #recommendation-panel ul { padding-left: 20px; list-style-type: '👉 '; }
        button { background-color: #28a745; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; width: 100%; margin-top: 10px;}
        button:hover { background-color: #218838; }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>Student Well-being Scenario Planner</h1>
        <div class="content-grid">
            <div class="controls">
                <h3>Your Lifestyle Inputs</h3>
                <div class="slider-group">
                    <label>Study Time: <span id="studytime-val">2-5 hours/week</span></label>
                    <input type="range" id="studytime" min="1" max="4" value="2" oninput="updateSliderText(this.id, this.value)">
                </div>
                <div class="slider-group">
                    <label>Socializing (Go Out): <span id="goout-val">Average</span></label>
                    <input type="range" id="goout" min="1" max="5" value="3" oninput="updateSliderText(this.id, this.value)">
                </div>
                <div class="slider-group">
                    <label>Alcohol Intake: <span id="total_alcohol-val">4 units/week</span></label>
                    <input type="range" id="total_alcohol" min="0" max="10" value="4" oninput="updateSliderText(this.id, this.value)">
                </div>
                 <div class="slider-group">
                    <label>Past Class Failures: <span id="failures-val">0</span></label>
                    <input type="range" id="failures" min="0" max="4" value="0" oninput="updateSliderText(this.id, this.value)">
                </div>
                <button onclick="runAnalysis()">Analyze My Lifestyle</button>
            </div>
            <div class="chart-container">
                <canvas id="scenarioChart"></canvas>
            </div>
        </div>
        <div id="recommendation-panel">
            <h3>Long-Term Impact & Recommendations</h3>
            <p><strong></p>
            <ul id="recommendation-list">
                <li>Adjust the sliders and click "Analyze" to see your personalized results.</li>
            </ul>
        </div>
    </div>
<script>
    let myChart;
    const studyLabels = { 1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h' };
    const gooutLabels = { 1: 'Very Low', 2: 'Low', 3: 'Average', 4: 'High', 5: 'Very High' };

    // Function to update the text next to the slider
    function updateSliderText(id, val) {
        let displayVal;
        if (id === 'studytime') displayVal = studyLabels[val];
        else if (id === 'goout') displayVal = gooutLabels[val];
        else if (id === 'total_alcohol') displayVal = `${val} units/week`;
        else displayVal = val;
        document.getElementById(`${id}-val`).innerText = displayVal;
    }

    // Main function to run the analysis when the button is clicked
    async function runAnalysis() {
        const baseInputs = {
            studytime: parseInt(document.getElementById('studytime').value),
            goout: parseInt(document.getElementById('goout').value),
            total_alcohol: parseInt(document.getElementById('total_alcohol').value),
            failures: parseInt(document.getElementById('failures').value)
        };

        const response = await fetch('/analyze-lifestyle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_inputs: baseInputs })
        });
        const result = await response.json();
        
        if (result.scenarios) {
            updateChartData(result.current_outcome, result.scenarios);
            updateRecommendationList(result.recommendations);
        }
    }
    
    // Function to update the chart's data with all scenarios
    function updateChartData(current, scenarios) {
        const colors = {
            studytime: 'rgba(0, 123, 255, 0.8)',
            goout: 'rgba(255, 193, 7, 0.8)',
            total_alcohol: 'rgba(220, 53, 69, 0.8)'
        };

        myChart.data.datasets = [
            // Plot the "what-if" scenarios
            { label: 'Impact of Study Time', data: scenarios.studytime, backgroundColor: colors.studytime, pointRadius: 5 },
            { label: 'Impact of Socializing', data: scenarios.goout, backgroundColor: colors.goout, pointRadius: 5 },
            { label: 'Impact of Alcohol', data: scenarios.total_alcohol, backgroundColor: colors.total_alcohol, pointRadius: 5 },
            // Plot the user's current position as a large, distinct point
            { label: 'Your Current Prediction', data: [current], backgroundColor: 'rgba(40, 167, 69, 1)', pointRadius: 10, pointHoverRadius: 12 }
        ];
        myChart.update();
    }
    
    // Function to update the recommendation list
    function updateRecommendationList(recs) {
        const listElement = document.getElementById('recommendation-list');
        listElement.innerHTML = ''; // Clear previous recommendations
        recs.forEach(rec => {
            const li = document.createElement('li');
            li.innerHTML = rec; // Use innerHTML to render the <strong> tags
            listElement.appendChild(li);
        });
    }

    // Initialize on page load
    window.onload = () => {
        const ctx = document.getElementById('scenarioChart').getContext('2d');
        myChart = new Chart(ctx, {
            type: 'scatter', data: { datasets: [] },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { 
                        title: { display: true, text: 'Predicted Health Score (1-5)', font: { size: 14 }},
                        grace: '5%' 
                    },
                    y: { 
                        title: { display: true, text: 'Predicted Final Grade (0-20)', font: { size: 14 }},
                        grace: '5%'
                    }
                },
                plugins: { legend: { position: 'bottom' }, tooltip: {
                    callbacks: { label: function(context) {
                        const label = context.dataset.label || '';
                        return `${label}: (Health: ${context.parsed.x.toFixed(1)}, Grade: ${context.parsed.y.toFixed(1)})`;
                    }}
                }}
            }
        });

        // Initialize all slider text
        ['studytime', 'goout', 'total_alcohol', 'failures'].forEach(id => updateSliderText(id, document.getElementById(id).value));
    };
</script>
</body>
</html>
"""

if __name__ == '__main__':
    import sys
    import multiprocessing
    import subprocess
    import threading
    import time
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == '--with-fake-scraper':
        flask_proc = multiprocessing.Process(target=run_flask)
        flask_proc.start()
        scraper_proc = subprocess.Popen([sys.executable, 'real_twitter_scraper_disabled.py'])
        def scraper_status_watcher(proc):
            while proc.poll() is None:
                print('Scraper is running...')
                time.sleep(5)
            print('Scraper has stopped.')
        status_thread = threading.Thread(target=scraper_status_watcher, args=(scraper_proc,), daemon=True)
        status_thread.start()
        flask_proc.join()
        if scraper_proc.poll() is None:
            print('Flask app exited. Terminating scraper...')
            scraper_proc.terminate()
        status_thread.join(timeout=1)
    else:
        run_flask()
