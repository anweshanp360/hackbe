const express = require('express');
const router = express.Router();
const { getTrials } = require('../controller/trial.controller');

router.get('/', async (req, res, next) => {
    try {
        await getTrials(req, res, next); // Assuming getTrials handles sending the response
    } catch (error) {
        // Log the error
        console.error('Error in getTrials route:', error);

        // Optionally, send an error response to the client
        // You might want to customize this based on your API's error handling strategy
        res.status(500).json({
            message: 'An error occurred while fetching trials.',
            error: error.message // Only send error.message in production, not the full stack
        });

        // Or, pass the error to the next error-handling middleware
        // next(error);
    }
});

router.post('/', async (req, res, next) => {
    try {
        await getTrials(req, res, next); // Assuming getTrials handles sending the response
    } catch (error) {
        // Log the error
        console.error('Error in getTrials route:', error);

        // Optionally, send an error response to the client
        // You might want to customize this based on your API's error handling strategy
        res.status(500).json({
            message: 'An error occurred while fetching trials.',
            error: error.message // Only send error.message in production, not the full stack
        });

        // Or, pass the error to the next error-handling middleware
        // next(error);
    }
});

module.exports = router;