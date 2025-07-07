const express = require('express');
const router = express.Router();
const patientController = require('../controller/patient.controller');

router.get('/', patientController.getPatients);
router.post('/', patientController.postPatients);

module.exports = router;
