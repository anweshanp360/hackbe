const patientService = require('../service/patient.service');

const getPatients = async (req, res) => {
  try {
    const filters = req.query;
    const patients = await patientService.getPatients(filters);
    res.status(200).json(patients);
  } catch (error) {
    console.error('❌ Controller error:', error);
    res.status(500).json({ message: 'Failed to fetch patient data' });
  }
};

const postPatients = async (req, res) => {
  try {
    const filters = req.query;
    const patients = await patientService.postPatient(filters);
    res.status(200).json(patients);
  } catch (error) {
    console.error('❌ Controller error:', error);
    res.status(500).json({ message: 'Failed to fetch patient data' });
  }
};


module.exports = {
  getPatients,
  postPatients
};
