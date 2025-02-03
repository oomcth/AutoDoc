import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import logging
from datetime import datetime
import torch
from typing import List, Dict, Optional, Tuple, Type
import subprocess
import os
from accelerate import Accelerator
from dataclasses import dataclass, fields, is_dataclass
import inspect
import sys
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
import os
from device_config import device


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
    nom_complet: Optional[str] = None
    consommation_tabac: Optional[str] = None
    consommation_alcool: Optional[str] = None
    antecedents_medicaux: Optional[str] = None
    allergies: Optional[str] = None
    intolerances_medicamenteuses: Optional[str] = None
    antecedents_chirurgicaux: Optional[str] = None
    antecedents_familiaux: Optional[str] = None
    antecedents_personnels_cancereux: Optional[str] = None
    antecedents_familiaux_cancereux: Optional[str] = None
    age_premiere_regle: Optional[str] = None
    nombre_enfants: Optional[str] = None
    nombre_grossesses: Optional[str] = None
    allaitement: Optional[str] = None
    date_menopause: Optional[str] = None
    traitement_hormonal_substitutif: Optional[str] = None
    taille_soutien_gorge: Optional[str] = None
    histoire_maladie_actuelle: Optional[str] = None
    poids_actuel: Optional[str] = None
    poids_habituel: Optional[str] = None
    taille: Optional[str] = None
    surface_corporelle: Optional[str] = None
    autres_notes: Optional[str] = None
    autre_infos: Optional[str] = None


class PatientInfoExtractor:
    """Extraction des informations patient depuis le texte"""
    def __init__(self):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
            use_auth_token=token
        )
        accelerator = Accelerator()
        self.model = accelerator.prepare(self.model)
        # self.model.to(device)

    def _generate_text(self, prompt: str, max_length: int = 1000) -> str:
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
        """
        Génère automatiquement un prompt basé sur la structure de la classe

        :param text: Texte de consultation à analyser
        :return: Prompt généré dynamiquement
        """
        # Générer des descriptions par défaut basées sur le nom du champ
        descriptions_champs = {
            field.name: ' '.join(
                mot.capitalize() for mot in 
                field.name.replace('_', ' ').split()
            ) for field in fields(PatientInfo())
        }

        # Construction du prompt
        prompt_sections = [
            "Extrayez les informations médicales suivantes du texte de consultation :",
            f"Texte: {text}",
            "\nInstructions :",
            "- Soyez précis et factuel",
            "- N'inventez pas d'informations",
            "- Si une information est manquante, indiquez 'aucun'",
            "\nFormat de réponse :"
        ]

        # Ajout dynamique des champs
        for champ, description in descriptions_champs.items():
            prompt_sections.append(f"{description}: Description du champ")

        prompt_sections.extend([
            "\nRègles supplémentaires :",
            "- Utilisez un format de texte clair et lisible",
            "- Ne pas utiliser de format JSON ou structuré",
            "- Soyez concis mais informatif",
            "\nVotre réponse :"
        ])
        print("\n".join(prompt_sections))
        response = self._generate_text("\n".join(prompt_sections))
        print("---" * 10)
        print("llm output : ", response)
        print("---" * 10)
        info = self._parse_patient_info(response)
        print("obtained info : ", info)

        # Génération des questions pour les informations manquantes
        missing_info = self.generer_questions_manquantes(info)
        return info, missing_info

    def _parse_patient_info(self, reponse: str) -> PatientInfo:
        """Parse la réponse du modèle pour extraire les informations structurées"""
        reponse = reponse.split("Votre réponse :")[1].strip()
        informations_extraites = {}

        for champ in fields(PatientInfo()):
            # Chercher une ligne commençant par le nom du champ
            ligne_correspondante = None
            for ligne in reponse.split('\n'):
                if ligne.strip().lower().startswith(champ.name.replace('_', ' ').lower() + ':'):
                    ligne_correspondante = ligne.split(':', 1)[1].strip()
                    break

            # Traitement de la valeur extraite
            if ligne_correspondante and ligne_correspondante.lower() not in ['', 'aucun', 'aucune']:
                # Gestion des types
                if champ.type == List[str]:
                    # Pour les listes, séparer par des virgules
                    informations_extraites[champ.name] = [
                        item.strip() for item in ligne_correspondante.split(',')
                        if item.strip()
                    ]
                else:
                    informations_extraites[champ.name] = ligne_correspondante

        return PatientInfo(**informations_extraites)

    def generer_questions_manquantes(self, infos: PatientInfo) -> List[str]:
        """
        Génère des questions pour les informations manquantes

        :param infos: Instance de dataclass avec les informations patient
        :return: Liste de questions pour les informations manquantes
        """
        questions_manquantes = []

        for champ in fields(infos):
            valeur = getattr(infos, champ.name)

            # Vérifier si la valeur est vide ou None
            if (valeur is None or
               (isinstance(valeur, list) and len(valeur) == 0) or
               (isinstance(valeur, str) and valeur.strip() == 'aucun')):

                # Formater le nom du champ pour la question
                nom_champ = ' '.join(mot.capitalize() for mot in champ.name.split('_'))

                # Générer une question contextuelle
                question = f"Pouvez-vous me donner plus d'informations sur {nom_champ.lower()} ?"
                questions_manquantes.append(question)

        return questions_manquantes


class AudioTranscriber:
    """Gestion de la transcription audio"""
    def __init__(self):
        self.model = whisper.load_model("base")
        # self.model.to(device)

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
        logger.info("Transcription terminée")

        # Extraction des informations patient
        logger.info("Extraction des informations patient...")
        patient_info, missing_patient_info = self.patient_extractor.extract_patient_info(transcript)
        logger.info("Extraction terminée")

        return patient_info, missing_patient_info

    def generer_document_final(self, patient: PatientInfo) -> str:
        """
        Génère un document final dynamique basé sur la structure du dataclass

        :param patient: Instance d'un dataclass avec les informations patient
        :return: Document final formatté
        """
        # Vérifier que c'est bien un dataclass
        if not is_dataclass(patient):
            raise ValueError("L'objet doit être un dataclass")

        # Sections du document
        sections = ["RÉSUMÉ DE CONSULTATION", ""]

        # Informations patient
        sections.append("INFORMATIONS PATIENT")
        sections.append("-" * 20)

        # Parcourir dynamiquement tous les champs
        for champ in fields(patient):
            valeur = getattr(patient, champ.name)

            # Formatage personnalisé selon le type de champ
            if valeur is None:
                libelle_valeur = f"Non renseigné(e)"
            elif isinstance(valeur, list):
                libelle_valeur = ', '.join(valeur) if valeur else "Aucun(e)"
            else:
                libelle_valeur = str(valeur)

            # Formater le nom du champ
            libelle_champ = ' '.join(mot.capitalize() for mot in champ.name.split('_'))

            sections.append(f"{libelle_champ}: {libelle_valeur}")

        # Ajouter une section générique pour toute information supplémentaire
        sections.append("\nAUTRES INFORMATIONS")
        sections.append("-" * 20)
        sections.append("Aucune information supplémentaire")

        return "\n".join(sections)


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
document = system.generer_document_final(patient_info)
print(document)
print(patient_info)
