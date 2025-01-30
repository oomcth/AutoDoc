import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import logging
from datetime import datetime
import torch
from typing import List, Dict, Optional, Tuple
import subprocess
import os
import sys
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
import os


token = "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"
model_name = "google/gemma-2-2b-it"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_executable(script_path):
    """Rend le script exécutable sur Unix"""
    if os.name != 'nt':  # Si ce n'est pas Windows
        os.chmod(script_path, 0o755)


def wait_key():
    """Attend qu'une touche soit pressée"""
    if os.name == 'nt':  # Windows
        return msvcrt.getch()
    else:  # Unix
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Désactive l'echo et le mode canonique
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def record_audio():
    start_script = "./start_record.sh"
    stop_script = "./end_record.sh"

    make_executable(start_script)
    make_executable(stop_script)

    try:
        print("Démarrage de l'enregistrement...")
        subprocess.run(['bash', start_script] if os.name != 'nt' else [start_script])

        print("Appuyez sur une touche pour arrêter l'enregistrement...")
        wait_key()

        print("Arrêt de l'enregistrement...")
        subprocess.run(['bash', stop_script] if os.name != 'nt' else [stop_script])

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        subprocess.run(['bash', stop_script] if os.name != 'nt' else [stop_script])


@dataclass
class PatientInfo:
    """Structure de données pour les informations patient"""
    nom: str
    prenom: str
    date_naissance: Optional[str] = None
    sexe: Optional[str] = None
    antecedents: List[str] = None
    allergies: List[str] = None
    medicaments: List[str] = None

    def is_complete(self) -> bool:
        """Vérifie si toutes les informations essentielles sont présentes"""
        required_fields = [self.nom, self.prenom, self.sexe]
        return all(required_fields)


class PatientInfoExtractor:
    """Extraction des informations patient depuis le texte"""
    def __init__(self):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="mps",
            use_auth_token=token
        )

    def _generate_text(self, prompt: str, max_length: int = 500) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise

    def extract_patient_info(self, text: str) -> Tuple[PatientInfo, List[str]]:
        """Extrait les informations patient et génère des questions si nécessaire"""
        prompt = f"""
        Extrayez les informations suivantes du texte de la consultation:
        Texte: {text}

        Format de réponse:
        Nom: [nom du patient]
        Prénom: [prénom du patient]
        Date de naissance: [date]
        Sexe: [M/F]
        Antécédents: [liste des antécédents]
        Allergies: [liste des allergies]
        Médicaments: [liste des médicaments]
        """

        response = self._generate_text(prompt)
        print("---" * 10)
        print("llm output : ", response)
        print("---" * 10)
        info = self._parse_patient_info(response)
        print("obtained info : ", info)

        # Génération des questions pour les informations manquantes
        missing_info = []
        if not info.nom or not info.prenom:
            missing_info.append("Quel est votre nom complet ?")
        if not info.date_naissance:
            missing_info.append("Quelle est votre date de naissance ?")
        if not info.sexe:
            missing_info.append("Quel est votre sexe ?")
        if not info.antecedents:
            missing_info.append("Avez-vous des antécédents médicaux ?")
        if not info.allergies:
            missing_info.append("Avez-vous des allergies ?")
        if not info.medicaments:
            missing_info.append("Prenez-vous des médicaments actuellement ?")

        return info, missing_info

    def _parse_patient_info(self, response: str) -> PatientInfo:
        """Parse la réponse du modèle pour extraire les informations structurées"""
        info = {
            "nom": "",
            "prenom": "",
            "date_naissance": None,
            "sexe": None,
            "antecedents": [],
            "allergies": [],
            "medicaments": []
        }

        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith("Nom:"):
                info["nom"] = line.replace("Nom:", "").strip()
            elif line.startswith("Prénom:"):
                info["prenom"] = line.replace("Prénom:", "").strip()
            elif line.startswith("Date de naissance:"):
                info["date_naissance"] = line.replace("Date de naissance:", "").strip()
            elif line.startswith("Sexe:"):
                sexe = line.replace("Sexe:", "").strip()
                if sexe.upper() in ['M', 'F']:
                    info["sexe"] = sexe.upper()
            elif line.startswith("Antécédents:"):
                antecedents = line.replace("Antécédents:", "").strip()
                if antecedents and antecedents.lower() != "aucun":
                    info["antecedents"] = [a.strip() for a in antecedents.split(',')]
            elif line.startswith("Allergies:"):
                allergies = line.replace("Allergies:", "").strip()
                if allergies and allergies.lower() != "aucune":
                    info["allergies"] = [a.strip() for a in allergies.split(',')]
            elif line.startswith("Médicaments:"):
                medicaments = line.replace("Médicaments:", "").strip()
                if medicaments and medicaments.lower() != "aucun":
                    info["medicaments"] = [m.strip() for m in medicaments.split(',')]

        return PatientInfo(**info)


class AudioTranscriber:
    """Gestion de la transcription audio"""
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path: str) -> str:
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Erreur lors de la transcription: {e}")
            raise


class MedicalTranscriptionSystem:
    """Système principal de transcription médicale"""
    def __init__(self):
        self.transcriber = AudioTranscriber()
        self.patient_extractor = PatientInfoExtractor()

    def process_consultation(self, audio_path: str
                             ) -> Tuple[PatientInfo,
                                        List[str]]:
        # Transcription
        logger.info("Début de la transcription...")
        transcript = self.transcriber.transcribe(audio_path)
        print(transcript)
        logger.info("Transcription terminée")

        # Extraction des informations patient
        logger.info("Extraction des informations patient...")
        patient_info, missing_patient_info = self.patient_extractor.extract_patient_info(transcript)
        logger.info("Extraction terminée")

        return patient_info, missing_patient_info

    def generate_final_document(self, patient: PatientInfo) -> str:
        """Génération du document final avec informations patient"""
        template = f"""
        RÉSUMÉ DE CONSULTATION
        ---------------------

        INFORMATIONS PATIENT
        -------------------
        Nom: {patient.nom}
        Prénom: {patient.prenom}
        Date de naissance: {patient.date_naissance or "Non renseignée"}
        Sexe: {patient.sexe or "Non renseigné"}

        Antécédents: {', '.join(patient.antecedents) if patient.antecedents else "Aucun"}
        Allergies: {', '.join(patient.allergies) if patient.allergies else "Aucune"}
        Médicaments: {', '.join(patient.medicaments) if patient.medicaments else "Aucun"}

        CONSULTATION
        -----------
        Motif de consultation:

        """
        return template


# Création du système
system = MedicalTranscriptionSystem()


# Traitement d'une consultation
record_audio()
patient_info, missing_info = system.process_consultation("output.wav")

# Si des informations sont manquantes
if missing_info:
    print("Questions à poser au patient:")
    for question in missing_info:
        print(f"- {question}")

# Document de sortie
document = system.generate_final_document(patient_info)
print(document)
print(patient_info)
