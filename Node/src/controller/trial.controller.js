const trialsService = require('../service/trial.service');

const getTrials = async (req, res) => {
  try {
    const trials = await trialsService.getAllTrials();
    res.status(200).json(trials);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const postTrials = async (req, res) => {
  try {
    const trials = await trialsService.createTrial();
    res.status(200).json(trials);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

module.exports = { getTrials,postTrials };