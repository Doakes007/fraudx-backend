CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    password VARCHAR(100),
    risk_level VARCHAR(20) DEFAULT 'low'
);

CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    balance NUMERIC DEFAULT 0
);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    from_id INTEGER,
    to_id INTEGER,
    amount NUMERIC,
    type VARCHAR(20),
    step INTEGER,
    oldbalanceOrg NUMERIC,
    newbalanceOrig NUMERIC,
    oldbalanceDest NUMERIC,
    newbalanceDest NUMERIC,
    is_fraud BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE user_devices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    device_id TEXT NOT NULL,
    location TEXT,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


