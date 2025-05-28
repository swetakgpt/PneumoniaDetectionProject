from flask import Flask, render_template_string, request
from flask_wtf.csrf import CSRFProtect
from markupsafe import Markup # Need this if you're building HTML manually
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_default_secret_for_testing_123')
app.config['WTF_CSRF_TIME_LIMIT'] = None 
csrf = CSRFProtect(app) # Initialize CSRFProtect

@app.route('/testform', methods=['GET', 'POST'])
def test_form():
    if request.method == 'POST':
        name = request.form.get('name')
        # CSRF validation will happen automatically in before_request by Flask-WTF
        return f"Form submitted! Name: {name}"

    # If csrf_token() global function indeed ONLY returns the raw token string
    # then we need to build the HTML ourselves for non-FlaskForm usage.
    # Flask-WTF expects the field to be named 'csrf_token' by default (app.config["WTF_CSRF_FIELD_NAME"])
    
    html_form = """
    <!DOCTYPE html>
    <html><head><title>CSRF Test (Manual Field)</title></head><body>
    <h1>CSRF Test Form (Manual Field Construction)</h1>
    <form method="POST" action="{{ url_for('test_form') }}">
        {# Generate the raw token value using the global function #}
        {% set raw_csrf_token_value = csrf_token() %}
        
        {# Manually construct the hidden input field #}
        <input id="csrf_token" name="csrf_token" type="hidden" value="{{ raw_csrf_token_value }}">
        
        <label for="name">Name:</label>
        <input type="text" id="name" name="name">
        <button type="submit">Submit</button>
    </form>
    <hr>
    <p>Raw token value used above: {{ raw_csrf_token_value }}</p>
    <p>View page source to confirm the hidden input field is correctly constructed.</p>
    </body></html>
    """
    return render_template_string(html_form)

if __name__ == '__main__':
    if not app.config['SECRET_KEY']: print("ERROR: SECRET_KEY is not set.")
    else: print(f"SECRET_KEY for test app is: {app.config['SECRET_KEY']}")
    app.run(debug=True, port=5001)