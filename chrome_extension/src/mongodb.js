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
});

const LogInCollection = mongoose.model('LogInCollection', logInSchema);

module.exports = LogInCollection;
