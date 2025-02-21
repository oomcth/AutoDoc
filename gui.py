import tkinter as tk
from tkinter import scrolledtext, Frame, Text
import subprocess
import os
import threading
from Medical_transcription import MedicalTranscriptionSystem, AudioTranscriber


class MessageBubble(tk.Frame):
    def __init__(self, master, message, is_user=True):
        super().__init__(master, bg="white")

        # Couleurs
        user_bg = "#DCF8C6"  # Vert clair pour l'utilisateur
        ai_bg = "#E8E8E8"    # Gris clair pour l'IA

        # Conteneur du message avec scrollbar
        self.text_frame = tk.Frame(self, bg=user_bg if is_user else ai_bg)
        self.text = tk.Text(self.text_frame, wrap=tk.WORD, width=80, height=1, 
                            font=("Arial", 12), bg=user_bg if is_user else ai_bg,
                            relief="flat", padx=10, pady=5, state="normal")

        # Scrollbar attachée au Text
        self.scrollbar = tk.Scrollbar(self.text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        # Insérer le message
        self.text.insert("1.0", message)
        self.text.configure(state='disabled')  # Empêcher l'édition

        # Ajuster automatiquement la hauteur
        self.text.update_idletasks()
        num_lines = int(self.text.index('end-1c').split('.')[0])
        self.text.configure(height=min(num_lines, 10))  # Max 10 lignes visibles

        # Pack des widgets
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.text_frame.pack(side="right" if is_user else "left", padx=10, pady=5, fill="x", expand=True)

        # Ajout du scrolling avec la molette de la souris
        self.text.bind("<Enter>", self._bind_mouse_scroll)
        self.text.bind("<Leave>", self._unbind_mouse_scroll)

    def _bind_mouse_scroll(self, event):
        self.text.bind_all("<MouseWheel>", self._on_mouse_scroll)
        self.text.bind_all("<Button-4>", self._on_mouse_scroll)  # Linux
        self.text.bind_all("<Button-5>", self._on_mouse_scroll)  # Linux

    def _unbind_mouse_scroll(self, event):
        self.text.unbind_all("<MouseWheel>")
        self.text.unbind_all("<Button-4>")
        self.text.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        if event.num == 4:  # Scroll up (Linux)
            self.text.yview_scroll(-1, "units")
        elif event.num == 5:  # Scroll down (Linux)
            self.text.yview_scroll(1, "units")
        else:  # Windows / macOS
            self.text.yview_scroll(-1 if event.delta > 0 else 1, "units")


class ChatFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="white")
        self.pack(fill="both", expand=True)

        # Canvas et scrollbar
        self.canvas = tk.Canvas(self, bg="white")
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        self.scrollable_frame.bind("<Configure>", lambda e: 
            self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack des widgets
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Ajuster la largeur dynamiquement
        self.bind("<Configure>", self.update_width)

        # Ajout du scroll avec la souris
        self.canvas.bind("<Enter>", self._bind_mouse_scroll)
        self.canvas.bind("<Leave>", self._unbind_mouse_scroll)

    def update_width(self, event=None):
        """ Ajuste la largeur du scrollable_frame à celle du canvas. """
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())

    def add_message(self, message, is_user=True):
        bubble = MessageBubble(self.scrollable_frame, message, is_user)
        bubble.pack(fill="x", padx=10, pady=5)
        self.canvas.yview_moveto(1)

    def _bind_mouse_scroll(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        self.canvas.bind_all("<Button-4>", self._on_mouse_scroll)  # Linux
        self.canvas.bind_all("<Button-5>", self._on_mouse_scroll)  # Linux

    def _unbind_mouse_scroll(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        if event.num == 4:  # Scroll up (Linux)
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Scroll down (Linux)
            self.canvas.yview_scroll(1, "units")
        else:  # Windows / macOS
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


class AudioChatInterface:
    def __init__(self, root, start_script, stop_script):
        self.root = root
        self.root.title("Interface de Chat Audio")

        # Chemins des scripts
        self.start_script = start_script
        self.stop_script = stop_script

        # Variables pour l'enregistrement
        self.recording = False

        # Configuration du transcriptor
        self.first_input = True
        self.transcriptor = MedicalTranscriptionSystem()
        self.transcriber = AudioTranscriber()
        self.audio_path = "output.wav"
        self.transcript = ""

        # Configuration de l'interface
        self.setup_gui()

    def setup_gui(self):
        # Zone de chat
        self.chat_frame = ChatFrame(self.root)
        self.chat_frame.grid(row=0, column=0, columnspan=3, padx=10,
                             pady=10, sticky="nsew")

        # Configuration du grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Boutons
        button_frame = Frame(self.root)
        button_frame.grid(row=1, column=0, columnspan=3, pady=5)

        self.record_button = tk.Button(button_frame,
                                       text="🎤 Enregistrer",
                                       command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame,
                                     text="✅ Garder",
                                     command=self.save_recording,
                                     state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.delete_button = tk.Button(button_frame,
                                       text="❌ Supprimer",
                                       command=self.delete_recording,
                                       state='disabled')
        self.delete_button.pack(side=tk.LEFT, padx=5)

        self.end_button = tk.Button(button_frame,
                                    text="🔚 Terminer",
                                    command=self.end_conversation)
        self.end_button.pack(side=tk.LEFT, padx=5)

        # Zone de transcript
        transcript_frame = Frame(self.root)
        transcript_frame.grid(row=2, column=0, columnspan=3,
                              padx=10, pady=5, sticky="ew")

        # Label pour le transcript
        transcript_label = tk.Label(transcript_frame, text="Transcript:",
                                    anchor="w")
        transcript_label.pack(side=tk.LEFT, padx=5)

        # Zone de texte éditable pour le transcript
        self.transcript_text = tk.Text(transcript_frame, height=10,
                                       wrap=tk.WORD)
        self.transcript_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Bouton pour mettre à jour le transcript
        self.update_button = tk.Button(transcript_frame,
                                       text="↻ Mettre à jour",
                                       command=self.update_transcript)
        self.update_button.pack(side=tk.RIGHT, padx=5)

        # Indicateur d'état
        self.status_label = tk.Label(self.root, text="Prêt à enregistrer")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.config(text="⏹️ Arrêter")
            self.status_label.config(text="Enregistrement en cours...")
            self.recording_thread = threading.Thread(target=self.start_recording)
            self.recording_thread.start()
        else:
            self.stop_recording()
            self.recording = False
            self.record_button.config(text="🎤 Enregistrer")
            self.status_label.config(text="Enregistrement terminé")
            self.save_button.config(state='normal')
            self.delete_button.config(state='normal')

    def start_recording(self):
        try:
            subprocess.run(['bash', self.start_script] if os.name != 'nt' else [self.start_script])
        except Exception as e:
            self.status_label.config(text=f"Erreur: {str(e)}")
            self.recording = False
            self.record_button.config(text="🎤 Enregistrer")

    def stop_recording(self):
        try:
            subprocess.run(['bash', self.stop_script] if os.name != 'nt' else [self.stop_script])
            self.transcript = self.transcriber.transcribe(self.audio_path)
            # Mettre à jour la zone de transcript
            self.transcript_text.delete('1.0', tk.END)
            self.transcript_text.insert('1.0', self.transcript)
        except Exception as e:
            self.status_label.config(text=f"Erreur: {str(e)}")

    def update_transcript(self):
        # Mettre à jour la variable transcript avec le contenu de la zone de texte
        self.transcript = self.transcript_text.get('1.0', tk.END).strip()
        self.status_label.config(text="Transcript mis à jour")

    def save_recording(self):
        # Utiliser le transcript mis à jour
        transcript = self.transcript

        # Ajouter les messages dans le chat
        self.chat_frame.add_message(transcript, is_user=True)

        self.patient_info, missing_info = self.transcriptor.process_consultation(transcript)
        response = self.transcriptor.generate_missing_data_prompt(missing_info)

        self.chat_frame.add_message(response, is_user=False)

        self.reset_interface()

    def end_conversation(self):
        self.chat_frame.add_message("Conversation terminée", is_user=False)
        self.root.quit()  # Ferme l'application
        document = self.transcriptor.generer_document_final(self.patient_info)
        print(document)
        print(self.patient_info)

    def delete_recording(self):
        self.transcript = ""
        self.transcript_text.delete('1.0', tk.END)
        self.reset_interface()

    def reset_interface(self):
        self.save_button.config(state='disabled')
        self.delete_button.config(state='disabled')
        self.status_label.config(text="Prêt à enregistrer")


def main():
    start_script = "./start_record.sh"
    stop_script = "./end_record.sh"
    root = tk.Tk()
    app = AudioChatInterface(root, start_script, stop_script)
    root.mainloop()


if __name__ == "__main__":
    main()
