const express = require('express');
const cors = require('cors'); // Import cors
const bcrypt = require('bcryptjs');
const app = express();
const LogInCollection = require('./mongodb');
const port = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors());

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Remove views and static file serving since we don't need them
// app.set('view engine', 'hbs');
// app.set('views', templatePath);
// app.use(express.static(publicPath));

// API Endpoint for Signup
app.post('/api/signup', async (req, res) => {
    const { email, password } = req.body;

    try {
        const existingUser = await LogInCollection.findOne({ email });

        if (existingUser) {
            return res.status(400).json({ message: 'User already exists' });
        } else {
            const hashedPassword = await bcrypt.hash(password, 10);
            const data = { email, password: hashedPassword };
            await LogInCollection.create(data);
            return res.status(201).json({ message: 'Signup successful' });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Error during signup' });
    }
});

// API Endpoint for Login
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;

    try {
        const user = await LogInCollection.findOne({ email });

        if (user && (await bcrypt.compare(password, user.password))) {
            // For simplicity, we won't generate a JWT or session here
            return res.status(200).json({ message: 'Login successful' });
        } else {
            return res.status(400).json({ message: 'Incorrect email or password' });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Error during login' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
