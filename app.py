from flask import Flask
from flask_cors import CORS

# Create Flask app instance
app = Flask(__name__)
CORS(app)

# Register route blueprints
from routes.users import user_bp
from routes.transactions import txn_bp
from routes.admin import admin_bp
from routes.auth import auth_bp

app.register_blueprint(user_bp, url_prefix='/api/user')
app.register_blueprint(txn_bp, url_prefix='/api/transaction')
app.register_blueprint(admin_bp, url_prefix='/api/admin')
app.register_blueprint(auth_bp, url_prefix='/api/auth')


print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(rule)


@app.route('/test', methods=['GET'])
def test():
    return {"message": "Flask is working!"}


# Run app
if __name__ == '__main__':
    app.run(debug=True)
    

