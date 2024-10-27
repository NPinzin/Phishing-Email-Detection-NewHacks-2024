// mongodb.js
const mongoose = require('mongoose');

mongoose
    .connect('mongodb://localhost:27017/')
    .then(() => {
        console.log('Mongoose connected');
    })
    .catch((e) => {
        console.error('Connection failed', e);
    });

const logInSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true,
        unique: true, // Ensure email uniqueness
    },
    password: {
        type: String,
        required: true,
    },
    verificationHistory: [
        {
            date: { type: Date, default: Date.now },
            emailData: {
                senderName: String,
                senderEmail: String,
                replyTo: String,
                subject: String,
                body: String,
            },
            prediction: Number,
        },
    ],
});

const LogInCollection = mongoose.model('LogInCollection', logInSchema);

module.exports = LogInCollection;


